import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Loader2, CheckCircle, XCircle, Download, Film } from 'lucide-react';
import { uploadVod, getJobStatus, getTimeline, downloadFile, JobResponse, TimelineData } from './api';
import Timeline from './components/Timeline';
import SkyBackground from './components/SkyBackground';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
      setJob(null);
      setTimeline(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mkv', '.avi', '.mov', '.webm'],
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const jobResponse = await uploadVod(file);
      setJob(jobResponse);

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobResponse.job_id);
          setJob(status);

          if (status.status === 'COMPLETED') {
            clearInterval(pollInterval);
            const timelineData = await getTimeline(jobResponse.job_id);
            setTimeline(timelineData);
          } else if (status.status === 'FAILED') {
            clearInterval(pollInterval);
            setError(status.error || 'Processing failed');
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDownload = async (fileType: 'timeline' | 'events' | 'summary') => {
    if (!job) return;
    try {
      await downloadFile(job.job_id, fileType);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  };

  const getStatusIcon = () => {
    if (!job) return null;
    switch (job.status) {
      case 'PENDING':
      case 'PROCESSING':
        return <Loader2 className="w-5 h-5 animate-spin text-c9-blue" />;
      case 'COMPLETED':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'FAILED':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getProgress = () => {
    if (!job?.progress) return 0;
    const { processed_frames, total_frames } = job.progress;
    return total_frames > 0 ? Math.round((processed_frames / total_frames) * 100) : 0;
  };

  return (
    <div className="min-h-screen">
      <SkyBackground />
      {/* Header */}
      <header className="bg-white/70 backdrop-blur-md border-b border-c9-cyan/30 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <Film className="w-8 h-8 text-c9-cyan" />
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">C9</span>
            <span className="text-c9-text"> VOD</span>
            <span className="text-c9-muted font-light"> Processor</span>
          </h1>
          <span className="text-c9-muted text-sm ml-2 tracking-widest uppercase">Cloud9 · Timeline Extractor</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        {!timeline && (
          <div className="mb-8">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all
                ${isDragActive 
                  ? 'border-c9-cyan bg-c9-cyan/10' 
                  : 'border-c9-cyan/40 hover:border-c9-cyan bg-white/60'
                }
                ${file ? 'border-c9-cyan bg-c9-cyan/10' : ''}
              `}
            >
              <input {...getInputProps()} />
              <Upload className={`w-12 h-12 mx-auto mb-4 ${file ? 'text-c9-cyan' : 'text-c9-muted'}`} />
              {file ? (
                <div>
                  <p className="text-lg font-medium text-c9-text">{file.name}</p>
                  <p className="text-sm text-c9-muted mt-1">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                  <p className="text-xs text-c9-muted/60 mt-2">Drop a different file to replace</p>
                </div>
              ) : (
                <div>
                  <p className="text-lg font-medium text-c9-text">Drop your VOD file here</p>
                  <p className="text-sm text-c9-muted mt-1">
                    or click to browse (MP4, MKV, AVI, MOV, WebM)
                  </p>
                </div>
              )}
            </div>

            {/* Upload Button */}
            <div className="mt-6">
              <button
                onClick={handleUpload}
                disabled={!file || isUploading || (job?.status === 'PROCESSING')}
                className="w-full py-3 px-6 bg-cyan-400 hover:bg-cyan-400 disabled:bg-c9-muted/30
                  disabled:cursor-not-allowed rounded-xl font-semibold tracking-wide transition-colors
                  text-white flex items-center justify-center gap-2"
              >
                {isUploading || job?.status === 'PROCESSING' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Process VOD
                  </>
                )}
              </button>
            </div>

            {/* Status */}
            {job && (
              <div className="mt-6 p-4 bg-white/70 backdrop-blur-md rounded-2xl border border-c9-cyan/40">
                <div className="flex items-center gap-3 mb-2">
                  {getStatusIcon()}
                  <span className="font-medium capitalize text-c9-text">{job.status.toLowerCase()}</span>
                  <span className="text-c9-muted text-sm ml-auto">
                    Job: {job.job_id.slice(0, 8)}...
                  </span>
                </div>
                {job.status === 'PROCESSING' && job.progress && (
                  <div className="mt-3">
                    <div className="flex justify-between text-sm text-c9-muted mb-1">
                      <span>
                        Frame {job.progress.processed_frames.toLocaleString()} / {job.progress.total_frames.toLocaleString()}
                      </span>
                      <span>{getProgress()}%</span>
                    </div>
                    <div className="h-3 bg-c9-cyan/15 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-c9-blue to-c9-light transition-all duration-300"
                        style={{ width: `${getProgress()}%` }}
                      />
                    </div>
                    <div className="mt-2 flex justify-between text-xs text-c9-muted">
                      <span>{job.progress.events_detected} events detected</span>
                      <span>~{Math.round((job.progress.total_frames - job.progress.processed_frames) / 30 / 60)} min remaining</span>
                    </div>
                  </div>
                )}
                {job.message && (
                  <p className="text-sm text-c9-muted mt-2">{job.message}</p>
                )}
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-300 rounded-2xl">
                <p className="text-red-600">{error}</p>
              </div>
            )}
          </div>
        )}

        {/* Timeline Display */}
        {timeline && (
          <div>
            {/* Download Buttons */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-bold text-c9-text">Timeline Results</h2>
                <p className="text-c9-muted text-sm">
                  {timeline.filename} • {timeline.total_kills} kills detected • {timeline.rounds.length} rounds
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => handleDownload('timeline')}
                  className="flex items-center gap-2 px-4 py-2 bg-white/80 hover:bg-white border border-c9-cyan/40 hover:border-c9-cyan text-c9-text rounded-xl transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Timeline JSON
                </button>
                <button
                  onClick={() => handleDownload('events')}
                  className="flex items-center gap-2 px-4 py-2 bg-white/80 hover:bg-white border border-c9-cyan/40 hover:border-c9-cyan text-c9-text rounded-xl transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Events JSON
                </button>
                <button
                  onClick={() => {
                    setTimeline(null);
                    setJob(null);
                    setFile(null);
                  }}
                  className="flex items-center gap-2 px-4 py-2 bg-c9-cyan hover:bg-cyan-400 text-white rounded-xl transition-colors font-semibold"
                >
                  Process Another
                </button>
              </div>
            </div>

            <Timeline data={timeline} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
