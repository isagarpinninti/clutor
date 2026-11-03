//! Remote control: a phone drives the desktop app over an end-to-end encrypted channel. The channel is the same whether the phone is on the same network or goes through a
//! relay you host, because the relay only forwards bytes it cannot read.

pub mod client;
pub mod commands;
pub mod host;
pub mod secure;
pub mod store;
