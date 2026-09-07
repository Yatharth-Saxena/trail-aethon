/**
 * AudioRecorder - Pure Web Audio API 16kHz 16-bit Mono PCM WAV Recorder
 * Captures clean, uncompressed WAV audio directly in any browser for transmission
 * to backend /api/voice/transcribe.
 */
export class AudioRecorder {
  constructor() {
    this.audioCtx = null;
    this.stream = null;
    this.scriptNode = null;
    this.source = null;
    this.pcmChunks = [];
    this.isRecording = false;
  }

  async start() {
    this.pcmChunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new AudioContextClass({ sampleRate: 16000 });
    this.source = this.audioCtx.createMediaStreamSource(this.stream);
    // 4096 sample buffer size
    this.scriptNode = this.audioCtx.createScriptProcessor(4096, 1, 1);

    this.scriptNode.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      this.pcmChunks.push(new Float32Array(channelData));
    };

    this.source.connect(this.scriptNode);
    this.scriptNode.connect(this.audioCtx.destination);
    this.isRecording = true;
  }

  stop() {
    this.isRecording = false;
    if (this.source) {
      try { this.source.disconnect(); } catch (e) {}
    }
    if (this.scriptNode) {
      try { this.scriptNode.disconnect(); } catch (e) {}
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    if (this.audioCtx) {
      try { this.audioCtx.close(); } catch (e) {}
    }

    // Merge PCM chunks
    const totalLen = this.pcmChunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of this.pcmChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Encode as 16-bit PCM WAV (16,000 Hz, 1 channel)
    const wavBuffer = new ArrayBuffer(44 + merged.length * 2);
    const view = new DataView(wavBuffer);

    const writeString = (v, off, str) => {
      for (let i = 0; i < str.length; i++) {
        v.setUint8(off + i, str.charCodeAt(i));
      }
    };

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + merged.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true); // NumChannels (1 = Mono)
    view.setUint32(24, 16000, true); // SampleRate (16000)
    view.setUint32(28, 32000, true); // ByteRate (16000 * 1 * 2)
    view.setUint16(32, 2, true); // BlockAlign (1 * 2)
    view.setUint16(34, 16, true); // BitsPerSample (16)
    writeString(view, 36, "data");
    view.setUint32(40, merged.length * 2, true);

    // Write samples
    let idx = 44;
    for (let i = 0; i < merged.length; i++) {
      const s = Math.max(-1, Math.min(1, merged[i]));
      view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      idx += 2;
    }

    return new Blob([view], { type: "audio/wav" });
  }
}
