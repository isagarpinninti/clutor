mod engine;
mod models;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .manage(engine::EngineState::default())
        .invoke_handler(tauri::generate_handler![
            engine::engine_open,
            engine::engine_op,
            engine::engine_close,
            models::system_info,
            models::list_models,
            models::model_catalog,
            models::reveal_models,
            models::download_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Clutor");
}
