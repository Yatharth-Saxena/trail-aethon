import React, { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';

const POSE_CONNECTIONS = [
  // Clavicle & Shoulders
  [11, 12],
  // Arms
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  // Hands
  [15, 17], [15, 19], [15, 21], [17, 19],
  [16, 18], [16, 20], [16, 22], [18, 20],
  // Flanks & Pelvis
  [11, 23], [12, 24], [23, 24],
  // Cross Torso Bracing Mesh
  [11, 24], [12, 23],
  // Legs
  [23, 25], [25, 27],
  [24, 26], [26, 28],
  // Feet
  [27, 29], [29, 31], [27, 31],
  [28, 30], [30, 32], [28, 32],
  // Head & Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [0, 11], [0, 12]
];

export default function Skeleton3D({ perceptionRef, style }) {
  const mountRef = useRef(null);
  const skeletonGroupRef = useRef(null);
  const spheresRef = useRef([]);
  const linesRef = useRef([]);
  const torsoMeshRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const gridRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Pre-create THREE.js objects to avoid recreation on each frame
  useEffect(() => {
    if (!mountRef.current) return;

    const width = 200;
    const height = 200;

    // Initialize scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Optimized camera position for better performance
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 20);
    camera.position.set(0, 0.1, 2.3);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Optimized renderer
    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Reduced for performance
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Subtle lighting for 3D meshes - optimized
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); // Reduced intensity
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0x00e6c8, 1.0); // Reduced intensity
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    // Optimized materials
    const jointMaterial = new THREE.MeshStandardMaterial({
      color: 0x00e6c8,
      emissive: 0x00a896,
      roughness: 0.25, // Slightly increased for performance
      metalness: 0.7   // Slightly decreased for performance
    });
    const majorJointMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x38bdf8,
      roughness: 0.15, // Slightly increased for performance
      metalness: 0.8   // Slightly decreased for performance
    });
    const boneMaterial = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      linewidth: 1.8,  // Slightly reduced for performance
      transparent: true,
      opacity: 0.8
    });
    const torsoMeshMaterial = new THREE.MeshStandardMaterial({
      color: 0x00e6c8,
      emissive: 0x005544,
      transparent: true,
      opacity: 0.2,    // Slightly reduced for performance
      side: THREE.DoubleSide,
      wireframe: false
    });

    // Create skeleton group
    const skeletonGroup = new THREE.Group();
    skeletonGroupRef.current = skeletonGroup;
    scene.add(skeletonGroup);

    // 1. Joint Spheres - Pre-allocate
    const pointsGroup = new THREE.Group();
    const spheres = [];
    for (let i = 0; i < 33; i++) {
      const isMajor = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i);
      const geo = new THREE.SphereGeometry(isMajor ? 0.032 : 0.018, 10, 10); // Reduced segments
      const mesh = new THREE.Mesh(geo, isMajor ? majorJointMaterial : jointMaterial);
      mesh.visible = false;
      pointsGroup.add(mesh);
      spheres.push(mesh);
    }
    skeletonGroup.add(pointsGroup);
    spheresRef.current = spheres;

    // 2. Bone Connection Lines - Pre-allocate
    const linesGroup = new THREE.Group();
    const lines = [];
    for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(6);
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const line = new THREE.Line(geo, boneMaterial);
      line.visible = false;
      linesGroup.add(line);
      lines.push(line);
    }
    skeletonGroup.add(linesGroup);
    linesRef.current = lines;

    // 3. Torso Polygonal Mesh Facets - Pre-allocate
    const torsoGeo = new THREE.BufferGeometry();
    // Increased vertices for better quality but still optimized
    const torsoVertices = new Float32Array(24); // Increased from 18 to 24
    torsoGeo.setAttribute('position', new THREE.BufferAttribute(torsoVertices, 3));
    const torsoMesh = new THREE.Mesh(torsoGeo, torsoMeshMaterial);
    torsoMesh.visible = false;
    skeletonGroup.add(torsoMesh);
    torsoMeshRef.current = torsoMesh;

    // 4. Ground Biometric Grid
    const grid = new THREE.GridHelper(1.8, 8, 0x00e6c8, 0x1e293b);
    grid.position.y = -0.95;
    scene.add(grid);
    gridRef.current = grid;

    // Animation loop with frame rate matching perception updates
    const animate = useCallback(() => {
      animationFrameRef.current = requestAnimationFrame(animate);

      // Reduced oscillation for better performance
      skeletonGroupRef.current.rotation.y = Math.sin(Date.now() * 0.0008) * 0.25;

      const p = perceptionRef?.current;
      const pose = p?.pose;

      // Early exit if no pose data
      if (!pose || (!pose.world_landmarks && !pose.landmarks)) {
        spheresRef.current.forEach(s => (s.visible = false));
        linesRef.current.forEach(l => (l.visible = false));
        torsoMeshRef.current.visible = false;
        return;
      }

      const hasWorld = pose.world_landmarks && pose.world_landmarks.length >= 33;
      const pts = hasWorld ? pose.world_landmarks : pose.landmarks;

      // Optimized coordinate getter
      const getCoord = (i) => {
        if (!pts[i]) return null;
        if (hasWorld) {
          return new THREE.Vector3(pts[i].x, -pts[i].y, -pts[i].z);
        } else {
          return new THREE.Vector3((pts[i].x - 0.5) * 1.5, -(pts[i].y - 0.5) * 1.5, 0);
        }
      };

      // Update joints
      const coords = [];
      for (let i = 0; i < 33; i++) {
        const c = getCoord(i);
        coords.push(c);
        if (c && (pts[i].visibility ?? 1) > 0.25) {
          spheresRef.current[i].position.copy(c);
          spheresRef.current[i].visible = true;
        } else {
          spheresRef.current[i].visible = false;
        }
      }

      // Update bones
      for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
        const [p1, p2] = POSE_CONNECTIONS[i];
        const c1 = coords[p1];
        const c2 = coords[p2];
        if (c1 && c2 && (pts[p1].visibility ?? 1) > 0.25 && (pts[p2].visibility ?? 1) > 0.25) {
          const pos = linesRef.current[i].geometry.attributes.position.array;
          pos[0] = c1.x; pos[1] = c1.y; pos[2] = c1.z;
          pos[3] = c2.x; pos[4] = c2.y; pos[5] = c2.z;
          linesRef.current[i].geometry.attributes.position.needsUpdate = true;
          linesRef.current[i].visible = true;
        } else {
          linesRef.current[i].visible = false;
        }
      }

      // Update torso
      const c11 = coords[11];
      const c12 = coords[12];
      const c23 = coords[23];
      const c24 = coords[24];

      if (c11 && c12 && c23 && c24) {
        const tPos = torsoMeshRef.current.geometry.attributes.position.array;
        tPos[0] = c11.x; tPos[1] = c11.y; tPos[2] = c11.z;
        tPos[3] = c12.x; tPos[4] = c12.y; tPos[5] = c12.z;
        tPos[6] = c24.x; tPos[7] = c24.y; tPos[8] = c24.z;

        tPos[9] = c11.x; tPos[10] = c11.y; tPos[11] = c11.z;
        tPos[12] = c24.x; tPos[13] = c24.y; tPos[14] = c24.z;
        tPos[15] = c23.x; tPos[16] = c23.y; tPos[17] = c23.z;
        torsoMeshRef.current.geometry.attributes.position.needsUpdate = true;
        torsoMeshRef.current.geometry.computeVertexNormals();
        torsoMeshRef.current.visible = true;
      } else {
        torsoMeshRef.current.visible = false;
      }

      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }, [perceptionRef]);

    // Start animation
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      if (mountRef.current && rendererRef.current.domElement) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      // Clean up references
      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      skeletonGroupRef.current = null;
      spheresRef.current = [];
      linesRef.current = [];
      torsoMeshRef.current = null;
      gridRef.current = null;
    };
  }, [perceptionRef]); // Re-run effect when perceptionRef changes

  return (
    <div
      style={{
        position: 'relative',
        width: 200,
        height: 200,
        background: 'rgba(8, 14, 22, 0.85)',
        border: '1px solid rgba(56, 189, 248, 0.35)',
        borderRadius: 10,
        overflow: 'hidden',
        backdropFilter: 'blur(12px)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.65)',
        ...style
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 6,
          left: 8,
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          fontSize: 9.5,
          fontFamily: 'var(--mono, monospace)',
          fontWeight: 700,
          letterSpacing: '0.06em',
          color: '#38bdf8',
          textTransform: 'uppercase',
          background: 'rgba(0, 0, 0, 0.45)',
          padding: '2px 6px',
          borderRadius: 4,
          border: '1px solid rgba(56, 189, 248, 0.2)'
        }}
      >
        <span
          style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: '#00e6c8',
            boxShadow: '0 0 6px #00e6c8'
          }}
        />
        3D SKELETAL MESH
      </div>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
