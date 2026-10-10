import { Canvas, useFrame } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function fade(t: number) {
  return t * t * (3 - 2 * t);
}

function hash3(ix: number, iy: number, iz: number) {
  const n = Math.sin(ix * 127.1 + iy * 311.7 + iz * 74.7) * 43758.5453123;
  return n - Math.floor(n);
}

function valueNoise(x: number, y: number, z: number) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = x - xi;
  const yf = y - yi;
  const zf = z - zi;
  const u = fade(xf);
  const v = fade(yf);
  const w = fade(zf);
  const n000 = hash3(xi, yi, zi);
  const n100 = hash3(xi + 1, yi, zi);
  const n010 = hash3(xi, yi + 1, zi);
  const n110 = hash3(xi + 1, yi + 1, zi);
  const n001 = hash3(xi, yi, zi + 1);
  const n101 = hash3(xi + 1, yi, zi + 1);
  const n011 = hash3(xi, yi + 1, zi + 1);
  const n111 = hash3(xi + 1, yi + 1, zi + 1);
  const nx00 = n000 * (1 - u) + n100 * u;
  const nx10 = n010 * (1 - u) + n110 * u;
  const nx01 = n001 * (1 - u) + n101 * u;
  const nx11 = n011 * (1 - u) + n111 * u;
  const nxy0 = nx00 * (1 - v) + nx10 * v;
  const nxy1 = nx01 * (1 - v) + nx11 * v;
  return nxy0 * (1 - w) + nxy1 * w;
}

function fbm(x: number, y: number, z: number) {
  let sum = 0;
  let amp = 0.5;
  let f = 1;
  for (let i = 0; i < 5; i++) {
    sum += amp * valueNoise(x * f, y * f, z * f);
    amp *= 0.5;
    f *= 2.05;
  }
  return sum;
}

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function makeBrainGeometry() {
  const geo = new THREE.IcosahedronGeometry(1, 28);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const cortex = new THREE.Color("#b7a196");
  const sulcus = new THREE.Color("#6d5c56");
  const occipital = new THREE.Color("#3ee0b2");

  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    x *= 0.92;
    y *= 0.82;
    z *= 1.12;

    const fissure = Math.exp(-((x * x) / 0.018)) * 0.22;
    x += Math.sign(x || 1) * 0.07;
    x *= 1 - fissure;

    const gyri = fbm(x * 2.8, y * 2.8, z * 2.8);
    const ridges = 0.09 * (gyri - 0.45);
    const p = new THREE.Vector3(x, y, z).normalize();
    x += p.x * ridges;
    y += p.y * ridges;
    z += p.z * ridges;

    y -= 0.04;
    pos.setXYZ(i, x, y, z);

    const posterior = smoothstep(-0.18, -0.72, z);
    const lowerBack = 1 - smoothstep(0.35, 0.85, y);
    const occ = posterior * lowerBack;
    const color = cortex.clone().lerp(sulcus, Math.max(0, 0.35 - gyri));
    color.lerp(occipital, occ);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

function BrainMesh() {
  const group = useRef<THREE.Group>(null);
  const occLight = useRef<THREE.PointLight>(null);
  const geometry = useMemo(() => makeBrainGeometry(), []);
  const cerebellum = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(0.28, 12);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const n = 1 + 0.12 * valueNoise(x * 6, y * 6, z * 6);
      pos.setXYZ(i, x * n * 1.15, y * n * 0.75, z * n);
    }
    geo.computeVertexNormals();
    return geo;
  }, []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (group.current) group.current.rotation.y = t * 0.18;
    if (occLight.current) occLight.current.intensity = 2.1 + Math.sin(t * 2.4) * 0.7;
  });

  return (
    <group ref={group}>
      <mesh geometry={geometry} castShadow>
        <meshStandardMaterial
          vertexColors
          roughness={0.55}
          metalness={0.08}
          emissive={new THREE.Color("#0c3d32")}
          emissiveIntensity={0.25}
        />
      </mesh>
      <mesh geometry={cerebellum} position={[0, -0.62, -0.72]}>
        <meshStandardMaterial color="#9d8b82" roughness={0.7} />
      </mesh>
      <mesh position={[0, -0.78, -0.18]} rotation={[0.5, 0, 0]}>
        <cylinderGeometry args={[0.09, 0.12, 0.42, 12]} />
        <meshStandardMaterial color="#8a766c" roughness={0.65} />
      </mesh>
      <Html position={[0.05, -0.12, -1.05]} center distanceFactor={6}>
        <div className="brain-label">
          Occipital lobe
          <small>visual cortex · SSVEP</small>
        </div>
      </Html>
      <pointLight ref={occLight} position={[0, -0.1, -1.2]} color="#3ee0b2" intensity={2.4} distance={2.4} />
    </group>
  );
}

export function Brain() {
  return (
    <div className="brain-stage" role="img" aria-label="3D brain with the occipital lobe highlighted">
      <Canvas camera={{ position: [2.15, 0.55, -1.55], fov: 38 }} gl={{ antialias: true }}>
        <color attach="background" args={["#070c0a"]} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[2.4, 2.2, 1.2]} intensity={1.15} />
        <directionalLight position={[-2, 0.6, -2]} intensity={0.35} color="#7ad0ff" />
        <BrainMesh />
        <OrbitControls enablePan={false} minDistance={2.2} maxDistance={5} autoRotate={false} />
      </Canvas>
      <p className="brain-caption">Drag to rotate. The green pole is where flicker-locked EEG is strongest.</p>
    </div>
  );
}
