import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

function colorForWord(word, index) {
  const seed = word.length * 17 + index * 29 + word.charCodeAt(0) * 3;
  const hue = seed % 360;
  return new THREE.Color(`hsl(${hue}, 85%, 62%)`);
}

export default function Visualizer({ text }) {
  const containerRef = useRef(null);

  const words = useMemo(() => {
    return text
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 90);
  }, [text]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x050816, 10, 40);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 2.2, 15);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xb4c6ff, 1.35);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(6, 8, 5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x59e3ff, 1.4);
    rimLight.position.set(-8, -2, -6);
    scene.add(rimLight);

    const root = new THREE.Group();
    scene.add(root);

    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.35, 3),
      new THREE.MeshStandardMaterial({
        color: 0x84f0ff,
        metalness: 0.75,
        roughness: 0.18,
        emissive: 0x113355,
        emissiveIntensity: 0.8,
      }),
    );
    root.add(core);

    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(2.4, 0.08, 16, 120),
      new THREE.MeshStandardMaterial({
        color: 0x7df9ff,
        metalness: 0.3,
        roughness: 0.1,
        emissive: 0x2d5cff,
        emissiveIntensity: 0.45,
      }),
    );
    halo.rotation.x = Math.PI / 2.7;
    root.add(halo);

    const tokenGroup = new THREE.Group();
    root.add(tokenGroup);

    const particles = [];
    const total = Math.max(words.length, 12);

    for (let i = 0; i < total; i += 1) {
      const word = words[i] || `token-${i}`;
      const radius = 0.16 + Math.min(word.length, 16) * 0.02;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 24, 24),
        new THREE.MeshStandardMaterial({
          color: colorForWord(word, i),
          metalness: 0.65,
          roughness: 0.22,
          emissive: colorForWord(word, i).clone().multiplyScalar(0.24),
          emissiveIntensity: 0.65,
        }),
      );

      const angle = i * 0.42;
      const height = (i - total / 2) * 0.22;
      const ring = 3.6 + (i % 5) * 0.18;

      mesh.position.set(
        Math.cos(angle) * ring,
        height,
        Math.sin(angle) * ring * 0.62,
      );

      tokenGroup.add(mesh);
      particles.push({ mesh, angle, height, ring });
    }

    const grid = new THREE.GridHelper(18, 24, 0x24507d, 0x102238);
    grid.position.y = -4.2;
    scene.add(grid);

    const onResize = () => {
      const nextWidth = container.clientWidth;
      const nextHeight = container.clientHeight;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };

    window.addEventListener("resize", onResize);

    let frame = 0;
    const animate = () => {
      frame += 1;
      const t = frame * 0.012;

      root.rotation.y = t * 0.32;
      root.rotation.x = Math.sin(t * 0.2) * 0.08;
      halo.rotation.z = t * 0.38;
      core.rotation.x += 0.008;
      core.rotation.y += 0.012;

      particles.forEach(({ mesh, angle, height, ring }, index) => {
        const wobble = Math.sin(t * 1.2 + index * 0.45) * 0.28;
        mesh.position.x = Math.cos(angle + t * 0.35) * (ring + wobble);
        mesh.position.y = height + Math.sin(t + index) * 0.16;
        mesh.position.z = Math.sin(angle + t * 0.35) * (ring * 0.62 + wobble * 0.42);
        mesh.rotation.x += 0.01;
        mesh.rotation.y += 0.015;
      });

      renderer.render(scene, camera);
      requestId = requestAnimationFrame(animate);
    };

    let requestId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(requestId);
      window.removeEventListener("resize", onResize);
      scene.traverse((object) => {
        if (object.geometry) {
          object.geometry.dispose();
        }
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach((material) => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
      renderer.dispose();
      container.innerHTML = "";
    };
  }, [words]);

  return <div ref={containerRef} className="visualizer-shell" />;
}

