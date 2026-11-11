//! The policy file (`clutor.policy.json`), read the same way the command line reads it (packages/core/src/node/policy.ts):
//! the path in CLUTOR_POLICY, else the file in the project folder. Unknown keys and wrong types are errors, so a typo cannot silently
//! switch protection off. It states what a company wants; it does not stop a person who can edit files on their own computer.
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Policy {
    /// Commands the agent runs are refused when this computer has no OS sandbox for them.
    pub require_sandbox: bool,
    /// "Auto-run checks", "Allow for this session" and saved allow rules are ignored: the person is always asked.
    pub forbid_auto_approve: bool,
    pub source: Option<String>,
}

const KEYS: [&str; 2] = ["requireSandbox", "forbidAutoApprove"];

pub fn parse(path: &str, text: &str) -> Result<Policy, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{path}: {e}"))?;
    let o = v.as_object().ok_or_else(|| format!("{path}: must be a JSON object"))?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) { return Err(format!("{path}: unknown key \"{k}\" (known: {})", KEYS.join(", "))); }
    }
    let flag = |k: &str| -> Result<bool, String> {
        match o.get(k) { None => Ok(false), Some(serde_json::Value::Bool(b)) => Ok(*b), Some(_) => Err(format!("{path}: \"{k}\" must be true or false")) }
    };
    Ok(Policy { require_sandbox: flag("requireSandbox")?, forbid_auto_approve: flag("forbidAutoApprove")?, source: Some(path.to_string()) })
}

pub fn load(root: &Path) -> Result<Policy, String> {
    let env = std::env::var("CLUTOR_POLICY").ok().filter(|s| !s.is_empty());
    let path = match &env { Some(p) => Path::new(p).to_path_buf(), None => root.join("clutor.policy.json") };
    if !path.exists() {
        if let Some(p) = env { return Err(format!("{p}: CLUTOR_POLICY names a file that does not exist")); }
        return Ok(Policy::default());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&path.display().to_string(), &text)
}

#[tauri::command]
pub fn project_policy(root: String) -> Result<Policy, String> {
    let root = std::path::PathBuf::from(&root).canonicalize().map_err(|e| format!("cannot open the project folder: {e}"))?;
    load(&root)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_both_switches_and_defaults_to_off() {
        assert_eq!(parse("p", "{}").unwrap(), Policy { require_sandbox: false, forbid_auto_approve: false, source: Some("p".into()) });
        let p = parse("p", r#"{"requireSandbox":true,"forbidAutoApprove":true}"#).unwrap();
        assert!(p.require_sandbox && p.forbid_auto_approve);
    }
    #[test]
    fn a_typo_or_wrong_type_is_an_error_not_a_silent_off() {
        assert!(parse("p", r#"{"requireSandbx":true}"#).unwrap_err().contains("unknown key"));
        assert!(parse("p", r#"{"requireSandbox":"yes"}"#).unwrap_err().contains("true or false"));
        assert!(parse("p", "[]").is_err());
        assert!(parse("p", "{not json").is_err());
    }
}
