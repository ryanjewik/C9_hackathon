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
    left: number;      // Score value
    right: number;     // Score value
    left_team: string; // Team code (e.g., "TH")
    right_team: string; // Team code (e.g., "PRX")
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

  const data = await response.json();
  // Normalize status casing from backend (backend returns lowercase)
  if (data && data.status && typeof data.status === 'string') {
    data.status = data.status.toUpperCase();
  }
  return data;
}

export async function getJobStatus(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/api/v1/vod/${jobId}/status`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get job status');
  }

  const data = await response.json();
  if (data && data.status && typeof data.status === 'string') {
    data.status = data.status.toUpperCase();
  }
  return data;
}

export async function getTimeline(jobId: string): Promise<TimelineData> {
  const response = await fetch(`${API_BASE}/api/v1/vod/${jobId}/timeline`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get timeline');
  }

  // Map backend TimelineResponse to frontend TimelineData shape
  const data = await response.json();

  // Backend may use `rounds_with_kills` or `rounds` depending on version
  const rounds = data.rounds || data.rounds_with_kills || [];

  // Extract team names from metadata.teams array or from round scores
  const teams = data.metadata?.teams || [];
  let leftTeam = teams[0] || '';
  let rightTeam = teams[1] || '';
  
  // If rounds have score info, use that for more accuracy
  if (rounds.length > 0 && rounds[0]?.score) {
    leftTeam = rounds[0].score.left_team || leftTeam;
    rightTeam = rounds[0].score.right_team || rightTeam;
  }

  // Map kill events to have proper display fields
  const mappedRounds = rounds.map((round: any) => ({
    ...round,
    start_ms: round.round_start_ms || round.start_ms || 0,
    kills: (round.kills || []).map((kill: any) => ({
      ...kill,
      timestamp_ms: kill.t_ms || kill.timestamp_ms,
      timestamp_display: kill.timestamp || kill.timestamp_display || '',
      killer_name: kill.killer || kill.killer_name,
      killer_team: kill.killer_color || kill.killer_team,
      victim_name: kill.victim || kill.victim_name,
      victim_team: kill.victim_color || kill.victim_team,
      weapon: kill.weapon || 'unknown',
      is_headshot: kill.headshot || kill.is_headshot || false,
    })),
  }));

  return {
    job_id: data.metadata?.vod_id || data.job_id || jobId,
    filename: data.metadata?.filename || (data.filename as any) || 'vod.mp4',
    duration_ms: data.metadata?.duration_ms || data.duration_ms || 0,
    resolution: data.metadata?.resolution || [0, 0],
    fps: data.metadata?.fps || data.fps || 30,
    rounds: mappedRounds,
    total_kills: data.metadata?.total_kills || data.total_kills || (Array.isArray(rounds) ? rounds.reduce((s: number, r: any) => s + (r.kills?.length || 0), 0) : 0),
    match_info: {
      left_team: leftTeam,
      right_team: rightTeam,
      map: data.metadata?.map || data.match_info?.map || 'Unknown Map',
    },
  } as TimelineData;
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
