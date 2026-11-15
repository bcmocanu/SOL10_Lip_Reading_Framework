import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, MouseEvent, SyntheticEvent } from "react";
import {
  FaceBox,
  getLipreadStatus,
  getMouthVideoUrl,
  getPreprocessStatus,
  JobStatusResponse,
  startLipread,
  startPreprocessWithUpload,
} from "../api/lipReading";
import "./LipReadingPage.css";

const MAX_SEGMENT_SECONDS = 15;
const MIN_SEGMENT_SECONDS = 0.05;

type SegmentPreview = {
  start: number;
  end: number;
  duration: number;
  key: number;
};

function roundSeconds(value: number) {
  return Math.round(value * 100) / 100;
}

function formatSeconds(value: number) {
  return `${value.toFixed(2)}s`;
}

function formatBox(box: FaceBox | null) {
  if (!box) return "No face selected";
  return `x=${Math.round(box.x)}, y=${Math.round(box.y)}, w=${Math.round(box.width)}, h=${Math.round(box.height)}`;
}

function buildSegmentMediaUrl(url: string, start: number, end: number) {
  return `${url}#t=${start.toFixed(3)},${end.toFixed(3)}`;
}

function getJobStatusText(label: string, job: JobStatusResponse | null) {
  if (!job) return "";

  if (job.status === "completed") return `${label} completed.`;
  if (job.status === "failed") return job.message ? `${label} failed: ${job.message}` : `${label} failed.`;
  if (job.status === "queued") return `${label} is queued.`;
  if (job.status === "running") return `${label} is running...`;

  return `${label}: ${job.status}`;
}

export default function LipReadingPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const segmentPreviewRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);

  const [startSeconds, setStartSeconds] = useState<number | null>(null);
  const [endSeconds, setEndSeconds] = useState<number | null>(null);
  const [currentVideoTime, setCurrentVideoTime] = useState<number>(0);
  const [duration, setDuration] = useState<number | null>(null);
  const [segmentPreview, setSegmentPreview] = useState<SegmentPreview | null>(null);
  const [segmentWarning, setSegmentWarning] = useState<string>("");

  const [frameImageUrl, setFrameImageUrl] = useState<string | null>(null);
  const [faceBox, setFaceBox] = useState<FaceBox | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const [preprocessJob, setPreprocessJob] = useState<JobStatusResponse | null>(null);
  const [mouthVideoUrl, setMouthVideoUrl] = useState<string | null>(null);
  const [lipreadJob, setLipreadJob] = useState<JobStatusResponse | null>(null);
  const [generatedText, setGeneratedText] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");

  const selectedDuration = useMemo(() => {
    if (startSeconds === null || endSeconds === null) return null;
    return Math.max(0, endSeconds - startSeconds);
  }, [startSeconds, endSeconds]);

  const playbackDurationFromStart = useMemo(() => {
    if (startSeconds === null) return null;
    return Math.max(0, currentVideoTime - startSeconds);
  }, [currentVideoTime, startSeconds]);

  const intervalIsValid =
    selectedDuration !== null && selectedDuration > 0 && selectedDuration <= MAX_SEGMENT_SECONDS;
  const canCropSegment = Boolean(file && previewUrl && intervalIsValid && startSeconds !== null && endSeconds !== null);
  const canExtractFrame = Boolean(segmentPreview && previewUrl);
  const canPreprocess = Boolean(file && segmentPreview && faceBox && intervalIsValid && preprocessJob?.status !== "running");
  const segmentMediaUrl = useMemo(() => {
    if (!previewUrl || !segmentPreview) return null;
    return buildSegmentMediaUrl(previewUrl, segmentPreview.start, segmentPreview.end);
  }, [previewUrl, segmentPreview]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!frameImageUrl || !canvasRef.current) return;

    const image = new Image();
    image.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
      drawFaceBox(ctx, faceBox);
    };
    image.src = frameImageUrl;
  }, [frameImageUrl, faceBox]);

  function getSafeDuration() {
    return duration !== null && Number.isFinite(duration) && duration > 0 ? duration : null;
  }

  function clampVideoTime(value: number) {
    const videoDuration = getSafeDuration();
    const rawValue = Number.isFinite(value) ? value : 0;
    const maxTime = videoDuration ?? Number.POSITIVE_INFINITY;
    return roundSeconds(Math.min(Math.max(rawValue, 0), maxTime));
  }

  function getSegmentValidationMessage(nextStart: number | null, nextEnd: number | null) {
    if (nextStart === null || nextEnd === null) return "";

    const nextDuration = nextEnd - nextStart;
    if (nextDuration <= 0) return "The stop time must be after the start time.";
    if (nextDuration > MAX_SEGMENT_SECONDS) {
      return `The selected segment is ${formatSeconds(nextDuration)}. Please keep it under ${MAX_SEGMENT_SECONDS}s.`;
    }

    return "";
  }

  function getCurrentPlaybackTime() {
    return clampVideoTime(videoRef.current?.currentTime ?? currentVideoTime);
  }

  function resetProcessingState() {
    setVideoId(null);
    setPreprocessJob(null);
    setMouthVideoUrl(null);
    setLipreadJob(null);
    setGeneratedText("");
  }

  function resetFrameSelection() {
    setFrameImageUrl(null);
    setFaceBox(null);
    resetProcessingState();
  }

  function resetSegmentPreviewAndFrame() {
    setSegmentPreview(null);
    resetFrameSelection();
  }

  function selectVideoFile(nextFile: File | null) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setFile(nextFile);
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null);
    setDuration(null);
    setStartSeconds(null);
    setEndSeconds(null);
    setCurrentVideoTime(0);
    setSegmentPreview(null);
    setSegmentWarning("");
    setErrorMessage("");
    resetFrameSelection();

    if (!nextFile && fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function clearWorkflow() {
    selectVideoFile(null);
    setIsDragActive(false);
    setIsDrawing(false);
    drawStartRef.current = null;
    setSegmentWarning("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectVideoFile(event.target.files?.[0] ?? null);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragActive(true);
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragActive(false);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragActive(false);

    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    if (!droppedFile) return;

    if (!droppedFile.type.startsWith("video/")) {
      setErrorMessage("Please drop a video file.");
      return;
    }

    selectVideoFile(droppedFile);
  }

  function handleClearSegmentMarkers() {
    setStartSeconds(null);
    setEndSeconds(null);
    setSegmentWarning("");
    setErrorMessage("");
    resetSegmentPreviewAndFrame();
  }

  function handleSetStart() {
    const nextStart = getCurrentPlaybackTime();
    const warning = getSegmentValidationMessage(nextStart, endSeconds);

    setStartSeconds(nextStart);
    setSegmentWarning(warning);
    setErrorMessage("");
    resetSegmentPreviewAndFrame();
  }

  function handleSetStop() {
    const nextEnd = getCurrentPlaybackTime();
    const warning = getSegmentValidationMessage(startSeconds, nextEnd);

    setEndSeconds(nextEnd);
    setSegmentWarning(warning);
    setErrorMessage("");
    resetSegmentPreviewAndFrame();
  }

  function handleVideoLoadedMetadata(event: SyntheticEvent<HTMLVideoElement>) {
    const nextDuration = event.currentTarget.duration;
    if (!Number.isFinite(nextDuration) || nextDuration <= 0) return;

    setDuration(roundSeconds(nextDuration));
    setCurrentVideoTime(roundSeconds(event.currentTarget.currentTime));
    setStartSeconds(null);
    setEndSeconds(null);
    setSegmentWarning("");
  }

  function handleMainVideoTimeChange(event: SyntheticEvent<HTMLVideoElement>) {
    setCurrentVideoTime(roundSeconds(event.currentTarget.currentTime));
  }

  function getSegmentPlaybackBounds(video: HTMLVideoElement) {
    if (!segmentPreview) return { start: startSeconds ?? 0, end: endSeconds ?? 0 };

    const mediaDuration = video.duration;
    const browserShowsTrimmedDuration = Number.isFinite(mediaDuration) && mediaDuration <= segmentPreview.duration + 0.5;

    if (browserShowsTrimmedDuration) {
      return { start: 0, end: Math.max(0, mediaDuration) };
    }

    return { start: segmentPreview.start, end: segmentPreview.end };
  }

  function handleSegmentLoadedMetadata(event: SyntheticEvent<HTMLVideoElement>) {
    const bounds = getSegmentPlaybackBounds(event.currentTarget);
    event.currentTarget.currentTime = bounds.start;
  }

  function handleSegmentPlay(event: SyntheticEvent<HTMLVideoElement>) {
    const bounds = getSegmentPlaybackBounds(event.currentTarget);

    if (event.currentTarget.currentTime < bounds.start || event.currentTarget.currentTime >= bounds.end) {
      event.currentTarget.currentTime = bounds.start;
    }
  }

  function handleSegmentTimeUpdate(event: SyntheticEvent<HTMLVideoElement>) {
    const bounds = getSegmentPlaybackBounds(event.currentTarget);

    if (event.currentTarget.currentTime >= bounds.end) {
      event.currentTarget.pause();
      event.currentTarget.currentTime = bounds.start;
    }
  }

  function handleCropSegment() {
    if (!canCropSegment || startSeconds === null || endSeconds === null || selectedDuration === null) return;

    setSegmentPreview({
      start: startSeconds,
      end: endSeconds,
      duration: selectedDuration,
      key: Date.now(),
    });
    setSegmentWarning("");
    setErrorMessage("");
    resetFrameSelection();
  }

  async function handleExtractFrame() {
    const video = videoRef.current;
    if (!video || !segmentPreview) return;

    setErrorMessage("");
    setFaceBox(null);
    resetProcessingState();

    const seekTo = Math.min(segmentPreview.start, Math.max(0, (duration ?? segmentPreview.start) - MIN_SEGMENT_SECONDS));

    await new Promise<void>((resolve, reject) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      const onError = () => {
        video.removeEventListener("error", onError);
        reject(new Error("The browser could not extract a frame from this video."));
      };
      video.addEventListener("seeked", onSeeked, { once: true });
      video.addEventListener("error", onError, { once: true });
      video.currentTime = seekTo;
    });

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setErrorMessage("This browser cannot draw the selected video frame.");
      return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setFrameImageUrl(canvas.toDataURL("image/jpeg", 0.92));
  }

  function getCanvasPoint(event: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();

    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function drawFrameWithBox(nextBox: FaceBox | null) {
    const canvas = canvasRef.current;
    if (!canvas || !frameImageUrl) return;

    const image = new Image();
    image.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      drawFaceBox(ctx, nextBox);
    };
    image.src = frameImageUrl;
  }

  function drawFaceBox(ctx: CanvasRenderingContext2D, box: FaceBox | null) {
    if (!box) return;

    ctx.save();
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = Math.max(3, ctx.canvas.width / 250);
    ctx.shadowBlur = 12;
    ctx.shadowColor = "rgba(0, 229, 255, 0.75)";
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = "rgba(0, 229, 255, 0.16)";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.restore();
  }

  function handleMouseDown(event: MouseEvent<HTMLCanvasElement>) {
    const point = getCanvasPoint(event);
    if (!point) return;
    drawStartRef.current = point;
    setIsDrawing(true);
  }

  function handleMouseMove(event: MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing || !drawStartRef.current) return;
    const point = getCanvasPoint(event);
    if (!point) return;

    const nextBox = normalizeBox(drawStartRef.current, point);
    drawFrameWithBox(nextBox);
  }

  function handleMouseUp(event: MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing || !drawStartRef.current) return;
    const point = getCanvasPoint(event);
    setIsDrawing(false);
    if (!point) return;

    const nextBox = normalizeBox(drawStartRef.current, point);
    if (nextBox.width < 8 || nextBox.height < 8) {
      setFaceBox(null);
      drawFrameWithBox(null);
      return;
    }

    setFaceBox(nextBox);
    resetProcessingState();
  }

  async function pollJob(
    getter: () => Promise<JobStatusResponse>,
    setter: (job: JobStatusResponse) => void,
  ) {
    for (let attempt = 0; attempt < 240; attempt += 1) {
      const job = await getter();
      setter(job);
      if (job.status === "completed" || job.status === "failed") return job;
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }

    throw new Error("The backend job did not finish in the expected time window.");
  }

  async function handlePreprocess() {
    if (!file || !faceBox || !segmentPreview) return;
    setErrorMessage("");
    setMouthVideoUrl(null);
    setGeneratedText("");
    setLipreadJob(null);

    try {
      const started = await startPreprocessWithUpload({
        file,
        start_seconds: segmentPreview.start,
        end_seconds: segmentPreview.end,
        face_box: faceBox,
      });

      const nextVideoId = started.video_id;
      setVideoId(nextVideoId);
      setPreprocessJob(started);

      const completed = await pollJob(
        () => getPreprocessStatus(nextVideoId),
        setPreprocessJob,
      );

      if (completed.status === "completed") {
        setMouthVideoUrl(`${getMouthVideoUrl(nextVideoId)}?t=${Date.now()}`);
      } else {
        setErrorMessage(completed.message ?? "Video preprocessing failed.");
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Video preprocessing failed.");
    }
  }

  async function handleLipread() {
    if (!videoId) return;
    setErrorMessage("");
    setGeneratedText("");

    try {
      const started = await startLipread(videoId);
      setLipreadJob(started);
      const completed = await pollJob(() => getLipreadStatus(videoId), setLipreadJob);

      if (completed.status === "completed") {
        setGeneratedText(completed.text ?? "");
      } else {
        setErrorMessage(completed.message ?? "Lip reading failed.");
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Lip reading failed.");
    }
  }

  return (
    <section className="lip-page">
      <div className="lip-hero-panel">
        <div className="lip-hero-content">
          <div className="lip-hero-copy">
            <p className="lip-hero-title">SOL10 Lip Reading</p>
            <p className="lip-subtitle">
              Select a video, choose a segment of up to 15 seconds, mark the face on the first frame,
              generate the mouth crop, then run the lip reading backend module.
            </p>
          </div>
          <button type="button" className="lip-clear-button" onClick={clearWorkflow}>
            Clear all
          </button>
        </div>
      </div>

      <div className="lip-workflow-grid">
        <div className="lip-card lip-card-main">
          <div className="lip-card-header">
            <div>
              <span className="lip-step">01</span>
              <h2>Video and segment</h2>
            </div>
          </div>

          <label
            className={`lip-upload-box${isDragActive ? " is-dragging" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileChange} />
            <span>Drop video here or click to choose</span>
            <small>{file ? `Selected file: ${file.name}` : "MP4, MOV, AVI, WEBM, or any browser-supported video format"}</small>
          </label>

          {previewUrl && (
            <video
              ref={videoRef}
              className="lip-video-preview"
              src={previewUrl}
              controls
              onLoadedMetadata={handleVideoLoadedMetadata}
              onTimeUpdate={handleMainVideoTimeChange}
              onSeeked={handleMainVideoTimeChange}
            />
          )}

          <div className="lip-marker-panel">
            <div className="lip-marker-grid">
              <div>
                <span>Current time</span>
                <strong>{formatSeconds(currentVideoTime)}</strong>
              </div>
              <div>
                <span>Start</span>
                <strong>{startSeconds !== null ? formatSeconds(startSeconds) : "Not set"}</strong>
              </div>
              <div>
                <span>Stop</span>
                <strong>{endSeconds !== null ? formatSeconds(endSeconds) : "Not set"}</strong>
              </div>
              <div>
                <span>Duration</span>
                <strong>{playbackDurationFromStart !== null ? formatSeconds(playbackDurationFromStart) : "Not set"}</strong>
              </div>
            </div>

            <div className="lip-marker-actions">
              <button type="button" onClick={handleClearSegmentMarkers} disabled={startSeconds === null && endSeconds === null}>
                Clear
              </button>
              <button type="button" onClick={handleSetStart} disabled={!previewUrl}>
                Set start
              </button>
              <button type="button" onClick={handleSetStop} disabled={!previewUrl}>
                Set stop
              </button>
            </div>

            {segmentWarning && <p className="lip-inline-warning">{segmentWarning}</p>}
          </div>

          <div className="lip-card-bottom-row">
            <div className="lip-actions">
              <button type="button" onClick={handleCropSegment} disabled={!canCropSegment}>
                Crop video
              </button>
            </div>

            <span className={intervalIsValid ? "lip-pill ok" : "lip-pill warn"}>
              {selectedDuration !== null && selectedDuration > 0
                ? `${formatSeconds(selectedDuration)} / max ${MAX_SEGMENT_SECONDS}s`
                : `Set segment / max ${MAX_SEGMENT_SECONDS}s`}
            </span>
          </div>
        </div>

        <div className="lip-card lip-card-main">
          <div className="lip-card-header">
            <div>
              <span className="lip-step">02</span>
              <h2>Cropped segment preview</h2>
            </div>
          </div>

          <p className="lip-help">
            Play the selected segment with the original audio, then extract the first frame for face marking.
          </p>

          {segmentMediaUrl && segmentPreview ? (
            <video
              key={segmentPreview.key}
              ref={segmentPreviewRef}
              className="lip-video-preview lip-segment-preview"
              src={segmentMediaUrl}
              controls
              onLoadedMetadata={handleSegmentLoadedMetadata}
              onPlay={handleSegmentPlay}
              onTimeUpdate={handleSegmentTimeUpdate}
            />
          ) : (
            <div className="lip-empty-state lip-segment-empty">The cropped segment preview will appear here.</div>
          )}

          <div className="lip-box-readout">
            Segment: {segmentPreview
              ? `${formatSeconds(segmentPreview.start)} - ${formatSeconds(segmentPreview.end)} (${formatSeconds(segmentPreview.duration)})`
              : "Not cropped yet"}
          </div>

          <div className="lip-actions">
            <button type="button" onClick={handleExtractFrame} disabled={!canExtractFrame}>
              Extract first frame
            </button>
          </div>
        </div>

        <div className="lip-card">
          <div className="lip-card-header">
            <div>
              <span className="lip-step">03</span>
              <h2>Face marking</h2>
            </div>
          </div>

          <p className="lip-help">
            Drag a rectangle over the selected face. These coordinates are used to select the matching tracked face.
          </p>

          <div className="lip-canvas-wrap">
            {frameImageUrl ? (
              <canvas
                ref={canvasRef}
                className="lip-face-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => setIsDrawing(false)}
              />
            ) : (
              <div className="lip-empty-state">The selected frame will appear here.</div>
            )}
          </div>

          <div className="lip-box-readout">{formatBox(faceBox)}</div>

          <div className="lip-actions">
            <button type="button" onClick={handlePreprocess} disabled={!canPreprocess}>
              Preprocess video
            </button>
          </div>

          {preprocessJob && <p className="lip-status">{getJobStatusText("Preprocessing", preprocessJob)}</p>}
        </div>

        <div className="lip-card lip-mouth-card">
          <div className="lip-card-header">
            <div>
              <span className="lip-step">04</span>
              <h2>Mouth video</h2>
            </div>
          </div>

          <p className="lip-help">
            This is the 96x96 mouth ROI generated by the preprocessing pipeline. Use it as the input for lip reading.
          </p>

          <div className="lip-mouth-body">
            <div className="lip-mouth-video-frame">
              {mouthVideoUrl ? (
                <video className="lip-mouth-video" src={mouthVideoUrl} controls />
              ) : (
                <div className="lip-empty-state lip-mouth-empty">The mouth video will appear here.</div>
              )}
            </div>

            <div className="lip-actions">
              <button
                type="button"
                onClick={handleLipread}
                disabled={!mouthVideoUrl || lipreadJob?.status === "running"}
              >
                Run lip reading
              </button>
            </div>

            {lipreadJob && <p className="lip-status">{getJobStatusText("Lip reading", lipreadJob)}</p>}
          </div>
        </div>

        <div className="lip-card lip-result-card">
          <div className="lip-card-header">
            <div>
              <span className="lip-step">05</span>
              <h2>Generated text</h2>
            </div>
          </div>

          <div className="lip-result-text">
            {generatedText || "The lip reading result will appear here."}
          </div>
        </div>
      </div>

      {errorMessage && <div className="lip-error">{errorMessage}</div>}
    </section>
  );
}

function normalizeBox(start: { x: number; y: number }, end: { x: number; y: number }): FaceBox {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  return { x, y, width, height };
}
