//! Video side. Start `server` first, then this.
//!
//! ```text
//! cargo run --example server
//! cargo run --example client
//! ```
//!
//! [`Client::connect`] returns one [`Session`]. This process opens a video
//! send track and accepts the server's audio track. The first video frame
//! is a keyframe. Later frames are deltas. [`SendTrack::control`] prints
//! bitrate updates and keyframe requests for that video track.
//!
//! Logs follow `RUST_LOG`. Unset, this prints `qrt=info`: the session and
//! its tracks. `RUST_LOG=qrt=debug` adds frames and bandwidth.
//! `RUST_LOG=qrt=trace` adds every datagram.

use anyhow::Result;
use bytes::Bytes;
use qrt::{Client, SendControl, engine::MediaKind};

const ADDR: &str = "127.0.0.1:7000";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("qrt=info")),
        )
        .init();

    let session = Client::connect(ADDR.parse()?, Default::default()).await?;

    println!("connected to {ADDR}");

    let video = session.open(MediaKind::Video).await?;
    println!("opened video stream {}", video.id());

    let audio = session.accept().await?;
    println!("accepted {:?} stream {}", audio.kind(), audio.id());

    let sending = async {
        for index in 0..3u32 {
            video
                .send(90_000 * index, index == 0, Bytes::from_static(b"pic"))
                .await?;
        }

        anyhow::Ok(())
    };

    let receiving = async {
        for _ in 0..3 {
            let frame = audio.recv().await?;
            println!(
                "frame stream {} timestamp {} keyframe {} payload {:?}",
                frame.stream_id, frame.timestamp, frame.keyframe, frame.payload
            );
        }

        anyhow::Ok(())
    };

    let controlling = async {
        loop {
            match video.control().await {
                Ok(SendControl::Rate(params)) => {
                    println!(
                        "rate stream {} bps {} rtt {:?} loss {:.3}",
                        video.id(),
                        params.target_bitrate_bps,
                        params.rtt,
                        params.loss_ratio
                    );
                }
                Ok(SendControl::Keyframe) => {
                    println!("keyframe requested on stream {}", video.id());
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

        let _ = video.close().await;
        let _ = audio.close().await;

        anyhow::Ok(())
    };

    let (done, ()) = tokio::join!(finished, controlling);

    done
}
