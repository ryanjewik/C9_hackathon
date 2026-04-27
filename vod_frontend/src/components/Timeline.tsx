import { TimelineData } from '../api';

interface TimelineProps {
  data: TimelineData;
}

export default function Timeline({ data }: TimelineProps) {
  return (
    <div className="p-6 bg-black/30 backdrop-blur-md rounded-xl border border-white/10">
      <p className="text-gray-300 text-sm">
        Timeline for <span className="text-white font-medium">{data.filename}</span> —{' '}
        {data.rounds.length} rounds, {data.total_kills} kills
      </p>
    </div>
  );
}
