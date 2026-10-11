import { Canvas, useFrame } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

function ImportedBrain({ object }: { object: THREE.Group }) {
  const group = useRef<THREE.Group>(null);
  const occLight = useRef<THREE.PointLight>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (group.current) group.current.rotation.y = t * 0.18;
    if (occLight.current) occLight.current.intensity = 2.1 + Math.sin(t * 2.4) * 0.7;
  });

  return (
    <group ref={group}>
      <primitive object={object} />
      <Html position={[0.05, -0.12, -0.9]} center distanceFactor={6}>
        <div className="brain-label">
          Occipital lobe
          <small>visual cortex · SSVEP</small>
        </div>
      </Html>
      <pointLight ref={occLight} position={[0, -0.1, -1]} color="#3ee0b2" intensity={2.4} distance={2.4} />
    </group>
  );
}

export function Brain() {
  const [brainObject, setBrainObject] = useState<THREE.Group | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadBrain() {
      try {
        const response = await fetch("Brain.obj");
        if (!response.ok) {
          throw new Error(`Could not load Brain.obj (${response.status}).`);
        }

        const object = new OBJLoader().parse(await response.text());
        const bounds = new THREE.Box3().setFromObject(object);
        const size = bounds.getSize(new THREE.Vector3());
        const largestDimension = Math.max(size.x, size.y, size.z);
        if (bounds.isEmpty() || !Number.isFinite(largestDimension) || largestDimension <= 0) {
          throw new Error("Brain.obj does not contain a valid 3D mesh.");
        }

        object.position.sub(bounds.getCenter(new THREE.Vector3()));
        object.scale.setScalar(1.8 / largestDimension);
        object.traverse((child) => {
          if (!(child instanceof THREE.Mesh)) return;
          if (!child.geometry.attributes.normal) child.geometry.computeVertexNormals();
          child.material = new THREE.MeshStandardMaterial({
            color: "#b7a196",
            roughness: 0.58,
            metalness: 0.04,
          });
        });

        if (!cancelled) setBrainObject(object);
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Could not load Brain.obj.");
        }
      }
    }

    void loadBrain();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="brain-stage" role="img" aria-label="3D Blender brain with the occipital lobe highlighted">
      <Canvas camera={{ position: [3.4, 0, -4.2], fov: 38 }} gl={{ antialias: true }}>
        <color attach="background" args={["#070c0a"]} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[2.4, 2.2, 1.2]} intensity={1.15} />
        <directionalLight position={[-2, 0.6, -2]} intensity={0.35} color="#7ad0ff" />
        {brainObject ? (
          <ImportedBrain object={brainObject} />
        ) : (
          <Html center>
            <div className="brain-label">{loadError ?? "Loading Blender brain…"}</div>
          </Html>
        )}
        <OrbitControls
          target={[0, -0.45, 0]}
          enablePan={false}
          minDistance={2.2}
          maxDistance={6.5}
          autoRotate={false}
        />
      </Canvas>
      <p className="brain-caption">
        Drag to rotate. The green pole is where flicker-locked EEG is strongest.
      </p>
    </div>
  );
}
