# AETHON — Mission Control Console

> **SEE. UNDERSTAND. ASSIST.**  
> Real-time AI perception, experiment assistance, and telemetry console for payload assembly workflows.

---

## Overview

**AETHON** is a high-performance aerospace mission control interface designed for human spaceflight payload operations. It combines:

- **Live Camera Perception**: Real-time object detection overlays (Person, Hands, Payload components, Controls).
- **Step-by-Step Guidance**: Interactive experiment progression tracker with status indicators.
- **Interactive 3D Moon Telemetry**: Three.js interactive moon rendered directly into mission control.
- **AI Mission Assistant**: Voice and text-driven mission support with real-time feedback and suggested queries.
- **Hardware Command Deck**: Streamlined experiment controls (`Start`, `Pause`, `Reset`, `End`) integrated into a unified glass panel.

---

## Tech Stack

- **Frontend**: [React 19](https://react.dev/), [Vite](https://vitejs.dev/)
- **Styling**: Vanilla CSS Design System with [Tailwind CSS v4](https://tailwindcss.com/)
- **3D Telemetry**: [Three.js](https://threejs.org/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Backend / Server**: Node.js & Express

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- `npm` or `pnpm`

### Installation

```bash
# Clone the repository
git clone https://github.com/Yatharth-Saxena/trail-aethon.git
cd trail-aethon

# Install dependencies
npm install
```

### Running Locally

```bash
# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
# Build the production bundle
npm run build

# Preview production build
npm run preview
```

---

## License

MIT
