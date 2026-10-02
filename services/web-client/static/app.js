// LiveKit WebRTC Voice Client
let currentRoom = null;
let audioContext = null;
let analyser = null;
let visualizerAnimationFrame = null;
let isMuted = false;

// DOM Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const roomInput = document.getElementById("roomInput");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnMute = document.getElementById("btnMute");
const muteIcon = document.getElementById("muteIcon");
const muteText = document.getElementById("muteText");
const speakingState = document.getElementById("speakingState");
const metricConnection = document.getElementById("metricConnection");
const transcriptBox = document.getElementById("transcriptBox");
const btnClear = document.getElementById("btnClear");
const remoteAudio = document.getElementById("remoteAudio");
const canvas = document.getElementById("visualizerCanvas");
const canvasCtx = canvas.getContext("2d");

// Helper: Add bubble to transcript
function addTranscriptBubble(sender, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${sender}-bubble`;
  bubble.textContent = text;
  transcriptBox.appendChild(bubble);
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

// Visualizer Setup
function startVisualizer(mediaStream) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }

  const source = audioContext.createMediaStreamSource(mediaStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    visualizerAnimationFrame = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = "rgba(12, 15, 23, 0.4)";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
      sum += dataArray[i];

      // Dynamic color gradient (Blue to Emerald)
      const r = Math.floor(59 + (16 - 59) * (i / bufferLength));
      const g = Math.floor(130 + (185 - 130) * (i / bufferLength));
      const b = Math.floor(246 + (129 - 246) * (i / bufferLength));
      canvasCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;

      canvasCtx.fillRect(
        x,
        (canvas.height - barHeight) / 2,
        barWidth,
        barHeight
      );

      x += barWidth + 2;
    }

    const avg = sum / bufferLength;
    if (avg > 15) {
      speakingState.textContent = "話しています (Listening / Speaking)";
      speakingState.style.borderColor = "#10b981";
      speakingState.style.color = "#10b981";
    } else {
      speakingState.textContent = "待機中 (Standby)";
      speakingState.style.borderColor = "rgba(255, 255, 255, 0.1)";
      speakingState.style.color = "#9ca3af";
    }
  }

  draw();
}

function stopVisualizer() {
  if (visualizerAnimationFrame) {
    cancelAnimationFrame(visualizerAnimationFrame);
    visualizerAnimationFrame = null;
  }
  if (canvasCtx) {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  }
  speakingState.textContent = "待機中";
}

// Connect to LiveKit Room
async function connectVoice() {
  const roomName = roomInput.value.trim() || "voice-room-1";

  // UI state
  statusDot.className = "status-dot connecting";
  statusText.textContent = "接続中...";
  btnConnect.disabled = true;

  try {
    // 1. Fetch JWT token from web client server
    const tokenRes = await fetch(`/api/token?room=${encodeURIComponent(roomName)}`);
    if (!tokenRes.ok) {
      throw new Error(`Token fetch failed: ${tokenRes.statusText}`);
    }
    const tokenData = await tokenRes.json();
    const token = tokenData.token;
    
    // Automatic LiveKit URL resolution:
    // If running in browser, connect to host IP on port 7880
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    const wsUrl = `${protocol}//${host}:7880`;

    // 2. Initialize LiveKit Room
    const { Room, RoomEvent, Track } = window.LivekitClient;
    currentRoom = new Room({
      adaptiveStream: true,
      dynacast: true,
      audioCaptureDefaults: {
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // 3. Setup event listeners
    currentRoom
      .on(RoomEvent.Connected, () => {
        statusDot.className = "status-dot connected";
        statusText.textContent = `接続中 (${roomName})`;
        metricConnection.textContent = "WebRTC Connected";
        metricConnection.className = "metric-val text-success";
        btnDisconnect.disabled = false;
        btnMute.disabled = false;
        addTranscriptBubble("system", `ルーム「${roomName}」に接続しました。マイクに向かって話しかけてください。`);
      })
      .on(RoomEvent.Disconnected, () => {
        handleDisconnectUI();
      })
      .on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        // AIエージェントの音声トラックを受信
        if (track.kind === Track.Kind.Audio) {
          track.attach(remoteAudio);
          addTranscriptBubble("system", `AIアシスタントが参加しました。`);
        }
      })
      .on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach();
      })
      .on(RoomEvent.DataReceived, (payload, participant) => {
        try {
          const str = new TextDecoder().decode(payload);
          const data = JSON.parse(str);
          if (data.type === "transcript") {
            addTranscriptBubble(data.sender || "agent", data.text);
          }
        } catch (e) {
          // ignore non-json data
        }
      });

    // 4. Connect to WebRTC room
    await currentRoom.connect(wsUrl, token);

    // 5. Enable Local Microphone & publish track
    await currentRoom.localParticipant.setMicrophoneEnabled(true);
    const micPublication = currentRoom.localParticipant.getTrackPublication(Track.Kind.Audio);
    if (micPublication && micPublication.track) {
      const mediaStream = new MediaStream([micPublication.track.mediaStreamTrack]);
      startVisualizer(mediaStream);
    }

  } catch (err) {
    console.error("Connection error:", err);
    alert(`接続に失敗しました: ${err.message}\nLiveKit Server (ws://${window.location.hostname}:7880) が稼働しているか確認してください。`);
    handleDisconnectUI();
  }
}

function handleDisconnectUI() {
  statusDot.className = "status-dot disconnected";
  statusText.textContent = "未接続";
  metricConnection.textContent = "Offline";
  metricConnection.className = "metric-val";
  btnConnect.disabled = false;
  btnDisconnect.disabled = true;
  btnMute.disabled = true;
  stopVisualizer();
  if (currentRoom) {
    currentRoom = null;
  }
}

async function disconnectVoice() {
  if (currentRoom) {
    await currentRoom.disconnect();
  }
  handleDisconnectUI();
}

async function toggleMute() {
  if (!currentRoom) return;
  isMuted = !isMuted;
  await currentRoom.localParticipant.setMicrophoneEnabled(!isMuted);
  if (isMuted) {
    muteIcon.textContent = "🔇";
    muteText.textContent = "解除";
    btnMute.style.background = "rgba(239, 68, 68, 0.2)";
  } else {
    muteIcon.textContent = "🎤";
    muteText.textContent = "ミュート";
    btnMute.style.background = "rgba(255, 255, 255, 0.08)";
  }
}

// Event Bindings
btnConnect.addEventListener("click", connectVoice);
btnDisconnect.addEventListener("click", disconnectVoice);
btnMute.addEventListener("click", toggleMute);
btnClear.addEventListener("click", () => {
  transcriptBox.innerHTML = "";
});
