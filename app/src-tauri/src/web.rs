//! Web access for the agent and for chat: search, and fetch a page as plain text. It runs here, not in the web view, so the app can refuse to be pointed at the user's own
//! network: every address a request would reach (including after each redirect) is resolved first and refused if it is private, loopback or link-local.

use regex::Regex;
use serde::Serialize;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;
use url::Url;

const MAX_BYTES: usize = 2 * 1024 * 1024;
const MAX_TEXT_CHARS: usize = 60_000;
const UA: &str = "Mozilla/5.0 (compatible; Clutor/0.4; +https://github.com/isagarpinninti/clutor)";

#[derive(Serialize, Debug, PartialEq)]
pub struct SearchHit { title: String, url: String, snippet: String }

#[derive(Serialize, Debug)]
pub struct Page { url: String, title: String, text: String, truncated: bool, status: u16 }

pub fn blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => blocked_v4(v),
        IpAddr::V6(v) => {
            if let Some(m) = v.to_ipv4_mapped() { return blocked_v4(&m); }
            let s = v.segments();
            v.is_loopback() || v.is_unspecified() || v.is_multicast() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 || *v == Ipv6Addr::LOCALHOST
        }
    }
}

fn blocked_v4(v: &Ipv4Addr) -> bool {
    let o = v.octets();
    v.is_loopback() || v.is_private() || v.is_link_local() || v.is_unspecified() || v.is_broadcast() || v.is_multicast() || v.is_documentation()
        || (o[0] == 100 && (64..128).contains(&o[1])) || o[0] == 0 || o[0] >= 240
}

/// One request, to an address that was checked. The connection is pinned to that address, so the name cannot resolve to something else a moment later.
async fn get_once(url: &Url, post_form: Option<&str>) -> Result<reqwest::Response, String> {
    if !matches!(url.scheme(), "http" | "https") { return Err("only http and https addresses can be opened".into()); }
    let host = url.host_str().ok_or("the address has no host")?.to_string();
    let port = url.port_or_known_default().ok_or("unknown port")?;
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), port)).await.map_err(|e| format!("could not look up {host}: {e}"))?.collect();
    if addrs.is_empty() { return Err(format!("{host} did not resolve")); }
    if addrs.iter().any(|a| blocked_ip(&a.ip())) { return Err("that address is on this computer or its local network, which the browser tools do not open".into()); }
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).timeout(Duration::from_secs(20)).user_agent(UA)
        .resolve(&host, addrs[0]).build().map_err(|e| e.to_string())?;
    let req = match post_form {
        Some(body) => client.post(url.clone()).header("content-type", "application/x-www-form-urlencoded").body(body.to_string()),
        None => client.get(url.clone()),
    };
    req.send().await.map_err(|e| format!("request failed: {e}"))
}

async fn fetch_follow(start: &str, post_form: Option<&str>) -> Result<(Url, reqwest::Response), String> {
    let mut url = Url::parse(start).map_err(|_| "that is not a valid address".to_string())?;
    for hop in 0..6 {
        let res = get_once(&url, if hop == 0 { post_form } else { None }).await?;
        if res.status().is_redirection() {
            let loc = res.headers().get("location").and_then(|v| v.to_str().ok()).ok_or("a redirect without a location")?;
            url = url.join(loc).map_err(|_| "a redirect to an invalid address".to_string())?;
            continue;
        }
        return Ok((url, res));
    }
    Err("too many redirects".into())
}

async fn read_capped(mut res: reqwest::Response) -> Result<(Vec<u8>, bool), String> {
    let mut out = Vec::new();
    while let Some(chunk) = res.chunk().await.map_err(|e| e.to_string())? {
        if out.len() + chunk.len() > MAX_BYTES { out.extend_from_slice(&chunk[..MAX_BYTES - out.len()]); return Ok((out, true)); }
        out.extend_from_slice(&chunk);
    }
    Ok((out, false))
}

#[tauri::command]
pub async fn web_fetch(url: String) -> Result<Page, String> {
    let (final_url, res) = fetch_follow(&url, None).await?;
    let status = res.status().as_u16();
    let ctype = res.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    if !(ctype.is_empty() || ctype.contains("text/") || ctype.contains("json") || ctype.contains("xml")) { return Err(format!("this is a {} file, not a page of text", ctype.split(';').next().unwrap_or("binary"))); }
    let (bytes, cut) = read_capped(res).await?;
    let body = String::from_utf8_lossy(&bytes).to_string();
    let (title, mut text) = if ctype.contains("html") || body.trim_start().starts_with('<') { html_to_text(&body) } else { (String::new(), body) };
    let mut truncated = cut;
    if text.chars().count() > MAX_TEXT_CHARS { text = text.chars().take(MAX_TEXT_CHARS).collect(); truncated = true; }
    Ok(Page { url: final_url.to_string(), title, text, truncated, status })
}

#[tauri::command]
pub async fn web_search(query: String) -> Result<Vec<SearchHit>, String> {
    let q = query.trim();
    if q.is_empty() { return Err("empty search".into()); }
    let form = format!("q={}", url::form_urlencoded::byte_serialize(q.chars().take(300).collect::<String>().as_bytes()).collect::<String>());
    let (_, res) = fetch_follow("https://html.duckduckgo.com/html/", Some(&form)).await?;
    if !res.status().is_success() { return Err(format!("the search service answered {}", res.status())); }
    let (bytes, _) = read_capped(res).await?;
    let hits = parse_search(&String::from_utf8_lossy(&bytes));
    if hits.is_empty() { return Err("no results could be read (the search service may have changed or refused the request)".into()); }
    Ok(hits)
}

fn decode_entities(s: &str) -> String {
    let named = s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'").replace("&amp;", "&");
    let num = Regex::new(r"&#(x?)([0-9a-fA-F]+);").unwrap();
    num.replace_all(&named, |c: &regex::Captures| {
        let n = if &c[1] == "x" { u32::from_str_radix(&c[2], 16).ok() } else { c[2].parse().ok() };
        n.and_then(char::from_u32).map(|ch| ch.to_string()).unwrap_or_default()
    }).to_string()
}

/// A page as readable text: the title, and the body without scripts, styles and markup, with block elements on their own lines.
pub fn html_to_text(html: &str) -> (String, String) {
    let title = Regex::new(r"(?is)<title[^>]*>(.*?)</title>").unwrap().captures(html).map(|c| decode_entities(c[1].trim())).unwrap_or_default();
    let mut s = Regex::new(r"(?is)<!--.*?-->").unwrap().replace_all(html, "").to_string();
    for tag in ["script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"] {
        s = Regex::new(&format!(r"(?is)<{tag}\b.*?</{tag}>")).unwrap().replace_all(&s, "").to_string();
    }
    s = Regex::new(r"(?i)<(br|/p|/div|/h[1-6]|/tr|/section|/article|/pre|/blockquote)[^>]*>").unwrap().replace_all(&s, "\n").to_string();
    s = Regex::new(r"(?i)<li[^>]*>").unwrap().replace_all(&s, "\n- ").to_string();
    s = Regex::new(r"(?s)<[^>]+>").unwrap().replace_all(&s, "").to_string();
    s = decode_entities(&s);
    let lines: Vec<String> = s.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    let mut out = String::new();
    let mut blank = 0;
    for l in lines {
        if l.is_empty() { blank += 1; if blank == 1 && !out.is_empty() { out.push('\n'); } } else { blank = 0; out.push_str(&l); out.push('\n'); }
    }
    (title, out.trim().to_string())
}

/// Results from the search service's plain-HTML page. The addresses it gives are redirect links; the real address is in their `uddg` parameter.
pub fn parse_search(html: &str) -> Vec<SearchHit> {
    let anchor = Regex::new(r#"(?is)<a\b([^>]*)>(.*?)</a>"#).unwrap();
    let href = Regex::new(r#"(?is)href\s*=\s*"([^"]*)""#).unwrap();
    let strip = Regex::new(r"(?s)<[^>]+>").unwrap();
    let mut hits: Vec<SearchHit> = Vec::new();
    for c in anchor.captures_iter(html) {
        let attrs = &c[1];
        let inner = decode_entities(strip.replace_all(&c[2], "").trim());
        if attrs.contains("result__a") {
            let Some(h) = href.captures(attrs) else { continue };
            let raw = decode_entities(&h[1]);
            let abs = if raw.starts_with("//") { format!("https:{raw}") } else { raw };
            let target = Url::parse(&abs).ok().and_then(|u| u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.to_string()).or_else(|| if u.host_str().map_or(true, |h| h.ends_with("duckduckgo.com")) { None } else { Some(u.to_string()) }));
            if let Some(t) = target { if t.starts_with("http") && !inner.is_empty() { hits.push(SearchHit { title: inner, url: t, snippet: String::new() }); } }
        } else if attrs.contains("result__snippet") {
            if let Some(last) = hits.last_mut() { if last.snippet.is_empty() { last.snippet = inner; } }
        }
    }
    hits.truncate(10);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_and_local_addresses_are_refused() {
        for a in ["127.0.0.1", "10.1.2.3", "192.168.0.5", "172.16.9.9", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fd00::1", "::ffff:127.0.0.1"] {
            assert!(blocked_ip(&a.parse().unwrap()), "{a} should be blocked");
        }
        for a in ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"] { assert!(!blocked_ip(&a.parse().unwrap()), "{a} should be allowed"); }
    }

    #[tokio::test]
    async fn local_targets_and_other_schemes_are_refused_before_any_request() {
        for u in ["http://127.0.0.1:8080/", "http://localhost/", "http://[::1]/", "http://169.254.169.254/latest/meta-data/"] {
            assert!(web_fetch(u.into()).await.is_err(), "{u}");
        }
        assert!(web_fetch("file:///etc/passwd".into()).await.unwrap_err().contains("http"));
        assert!(web_fetch("not a url".into()).await.is_err());
    }

    #[test]
    fn html_becomes_readable_text() {
        let (title, text) = html_to_text("<html><head><title>Hi &amp; bye</title><style>p{}</style></head><body><nav>menu</nav><h1>Head</h1><p>One <b>two</b></p><script>alert(1)</script><ul><li>a</li><li>b</li></ul>&#169; &lt;x&gt;</body></html>");
        assert_eq!(title, "Hi & bye");
        assert!(text.contains("Head\nOne two"));
        assert!(text.contains("- a\n- b"));
        assert!(text.contains("© <x>"));
        assert!(!text.contains("alert") && !text.contains("menu"));
    }

    #[test]
    fn search_results_are_parsed_and_their_redirects_unwrapped() {
        let html = r#"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">Example <b>A</b></a>
            <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The first &amp; best</a></div>
            <div class="result"><a class="result__a" href="https://other.org/b">Other</a><a class="result__snippet">Second</a></div>"#;
        let hits = parse_search(html);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], SearchHit { title: "Example A".into(), url: "https://example.com/a?x=1".into(), snippet: "The first & best".into() });
        assert_eq!(hits[1].url, "https://other.org/b");
    }

    /// Needs the network: `cargo test --lib web::tests::live -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn live() {
        let p = web_fetch("https://example.com/".into()).await.unwrap();
        println!("{} | {} | {}", p.url, p.title, p.text.chars().take(120).collect::<String>());
        assert!(p.title.contains("Example"));
        let hits = web_search("rust programming language".into()).await.unwrap();
        for h in hits.iter().take(3) { println!("{} -> {}", h.title, h.url); }
        assert!(!hits.is_empty());
    }
}
