import { AlertTriangle, RotateCcw } from "lucide-react";
import { Component } from "react";
import { cn } from "@/lib/utils";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-screen p-8 bg-[#050607] text-white">
          <div className="glass-panel flex flex-col items-center w-full max-w-2xl p-8 rounded-2xl">
            <AlertTriangle size={48} className="text-red-400 mb-6 flex-shrink-0" />
            <h2 className="text-xl mb-4 font-medium">An unexpected telemetry error occurred.</h2>
            <div className="p-4 w-full rounded bg-black/40 border border-white/10 overflow-auto mb-6">
              <pre className="text-sm text-white/60 whitespace-break-spaces">
                {this.state.error?.stack}
              </pre>
            </div>
            <button
              onClick={() => window.location.reload()}
              className={cn("control-button primary flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer")}
            >
              <RotateCcw size={16} />
              Reload Console
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
