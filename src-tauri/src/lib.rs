mod commands;
mod worker;

use std::sync::mpsc;

use tauri::{Emitter, Manager};

use commands::{WorkerCommand, WorkerHandle};

pub fn run() {
    let _ = dotenvy::dotenv();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();

            // Spawn the background worker thread.
            let handle = app.handle().clone();
            worker::spawn(cmd_rx, handle);

            // Spawn hotkey bridge thread.
            let hotkey_handle = app.handle().clone();
            std::thread::spawn(move || {
                let hotkey_rx = pulse::platform::hotkey::listen();
                let mut recording = false;
                loop {
                    match hotkey_rx.recv() {
                        Ok(()) => {
                            recording = !recording;
                            let event = if recording { "pulse:hotkey-start" } else { "pulse:hotkey-stop" };
                            if let Err(e) = hotkey_handle.emit(event, ()) {
                                eprintln!("[tauri] Failed to emit {}: {}", event, e);
                            }
                        }
                        Err(_) => break,
                    }
                }
            });

            // Manage worker handle as Tauri state.
            app.manage(WorkerHandle {
                cmd_tx: parking_lot::Mutex::new(cmd_tx),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            commands::start_recording,
            commands::stop_recording,
            commands::get_default_settings,
            commands::get_providers,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
