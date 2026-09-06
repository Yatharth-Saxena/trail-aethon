import { AlertCircle, Home } from "lucide-react";
import { useLocation } from "wouter";

export default function NotFound() {
  const [, setLocation] = useLocation();

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[#050607] text-[#f1f4f4] p-4">
      <div className="glass-panel w-full max-w-md p-8 text-center rounded-2xl flex flex-col items-center">
        <div className="relative mb-6">
          <div className="absolute inset-0 bg-red-500/20 rounded-full blur-md" />
          <AlertCircle className="relative h-14 w-14 text-red-400" />
        </div>

        <h1 className="text-4xl font-bold mb-2">404</h1>
        <h2 className="text-lg font-medium text-white/80 mb-3">Telemetry / Page Not Found</h2>
        <p className="text-sm text-white/50 mb-6 leading-relaxed">
          The requested mission view does not exist or has been relocated.
        </p>

        <button
          onClick={() => setLocation("/")}
          className="control-button primary flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg text-sm cursor-pointer"
        >
          <Home className="w-4 h-4" />
          Return to Mission Console
        </button>
      </div>
    </div>
  );
}
