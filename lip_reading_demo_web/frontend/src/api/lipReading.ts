import { API_BASE_URL, apiGet, apiPost, apiPostFormData } from "./client";

export type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PreprocessUploadRequest = {
  file: File;
  start_seconds: number;
  end_seconds: number;
  face_box: FaceBox;
};

export type JobStatus = "not_started" | "queued" | "running" | "completed" | "failed";

export type JobStatusResponse = {
  video_id: string;
  job_type: "preprocess" | "lipread";
  status: JobStatus;
  message?: string | null;
  result?: Record<string, unknown> | null;
  text?: string | null;
};

export async function startPreprocessWithUpload(
  payload: PreprocessUploadRequest,
): Promise<JobStatusResponse> {
  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("start_seconds", String(payload.start_seconds));
  formData.append("end_seconds", String(payload.end_seconds));
  formData.append("face_box", JSON.stringify(payload.face_box));

  return apiPostFormData<JobStatusResponse>("/api/videos/preprocess", formData);
}

export async function getPreprocessStatus(videoId: string): Promise<JobStatusResponse> {
  return apiGet<JobStatusResponse>(`/api/videos/${videoId}/preprocess-status`);
}

export function getMouthVideoUrl(videoId: string): string {
  return `${API_BASE_URL}/api/videos/${videoId}/mouth-video`;
}

export async function startLipread(videoId: string): Promise<JobStatusResponse> {
  return apiPost<JobStatusResponse>(`/api/videos/${videoId}/lipread`);
}

export async function getLipreadStatus(videoId: string): Promise<JobStatusResponse> {
  return apiGet<JobStatusResponse>(`/api/videos/${videoId}/lipread-status`);
}
