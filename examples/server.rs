//! Audio side. Start this, then `client` in another terminal.
//!
//! ```text
//! cargo run --example server
//! cargo run --example client
//! ```
//!
//! [`Server`] only accepts the TCP connection. The [`Session`] it returns
//! is the same type the client gets: this process opens an audio send track
//! and accepts the client's video track. Stream ids are printed; the
//! application does not choose them. [`SendTrack::control`] prints bitrate
//! updates for the audio track.
//!
//! Logs follow `RUST_LOG`. Unset, this prints `qrt=info`: the session and
//! its tracks. `RUST_LOG=qrt=debug` adds frames and bandwidth.
//! `RUST_LOG=qrt=trace` adds every datagram.

use anyhow::Result;
use bytes::Bytes;
use qrt::{SendControl, Server, engine::MediaKind};

const ADDR: &str = "127.0.0.1:7000";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("qrt=info")),
        )
        .init();

    let server = Server::bind(ADDR.parse()?, Default::default()).await?;
    println!("listening on {ADDR}");

    let session = server.accept().await?;

    let audio = session.open(MediaKind::Audio).await?;
    println!("opened audio stream {}", audio.id());

    let video = session.accept().await?;
    println!("accepted {:?} stream {}", video.kind(), video.id());

    let sending = async {
        for index in 0..3u32 {
            audio
                .send(48_000 * index, false, Bytes::from_static(b"opus"))
                .await?;
        }

        anyhow::Ok(())
    };

    let receiving = async {
        for _ in 0..3 {
            let frame = video.recv().await?;
            println!(
                "frame stream {} timestamp {} keyframe {} payload {:?}",
                frame.stream_id, frame.timestamp, frame.keyframe, frame.payload
            );
        }

        anyhow::Ok(())
    };

    let controlling = async {
        loop {
            match audio.control().await {
                Ok(SendControl::Rate(params)) => {
                    println!(
                        "rate stream {} bps {} rtt {:?} loss {:.3}",
                        audio.id(),
                        params.target_bitrate_bps,
                        params.rtt,
                        params.loss_ratio
                    );
                }
                Ok(SendControl::Keyframe) => {
                    println!("keyframe requested on stream {}", audio.id());
                }
                Err(_) => break,
            }
        }
    };

    let finished = async {
        let (sent, received) = tokio::join!(sending, receiving);
        sent?;
        received?;

        if let Ok(info) = session.info().await {
            println!("target {} bps rtt {:?}", info.target_bitrate_bps, info.rtt);
        }

        let _ = audio.close().await;
        let _ = video.close().await;

        anyhow::Ok(())
    };

    let (done, ()) = tokio::join!(finished, controlling);

    done
}
