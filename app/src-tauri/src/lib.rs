mod engine;
mod models;
mod project;
mod web;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(engine::EngineState::default())
        .manage(project::McpState::default())
        .invoke_handler(tauri::generate_handler![
            engine::engine_open,
            engine::engine_op,
            engine::engine_close,
            models::system_info,
            models::list_models,
            models::model_catalog,
            models::reveal_models,
            models::download_model,
            project::project_list,
            project::project_read,
            project::project_write,
            project::project_run,
            project::mcp_start,
            project::mcp_send,
            project::mcp_stop,
            web::web_fetch,
            web::web_search,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Clutor");
}
