// API Types

export interface JobResponse {
  job_id: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  message?: string;
  created_at: string;
  completed_at?: string;
  progress?: {
    processed_frames: number;
    total_frames: number;
    events_detected: number;
  };
  error?: string;
}

export interface KillEvent {
  timestamp_ms: number;
  timestamp_display: string;
  killer_name: string;
  killer_team: string;
  victim_name: string;
  victim_team: string;
  weapon: string;
  is_headshot: boolean;
}

export interface RoundData {
  round_number: number;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  winner?: string;
  kills: KillEvent[];
  score?: {
    left_team: number;
    right_team: number;
    left_team_name?: string;
    right_team_name?: string;
  };
}

export interface TimelineData {
  job_id: string;
  filename: string;
  duration_ms: number;
  resolution: [number, number];
  fps: number;
  rounds: RoundData[];
  total_kills: number;
  match_info?: {
    left_team?: string;
    right_team?: string;
    map?: string;
  };
}

// API functions

const API_BASE = '';

export async function uploadVod(file: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/v1/vod/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/api/v1/vod/${jobId}/status`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get job status');
  }

  return response.json();
}

export async function getTimeline(jobId: string): Promise<TimelineData> {
  const response = await fetch(`${API_BASE}/api/v1/vod/${jobId}/timeline`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get timeline');
  }

  return response.json();
}

export async function downloadFile(jobId: string, fileType: 'timeline' | 'events' | 'summary'): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/vod/${jobId}/download/${fileType}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to download file');
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${jobId}_${fileType}.json`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
