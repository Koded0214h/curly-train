import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

const PALETTE = [
  0x7cf9ff, 0x8f7cff, 0xff7ba8, 0x7bffa8, 0xffd87b,
  0xff9d7b, 0xb47bff, 0x7bddff, 0xffb07b,
];

function nodeColor(word, index) {
  const hash = [...word].reduce((a, c) => a * 31 + c.charCodeAt(0), index * 7);
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

function makeLabel(text, color) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 26px Inter, ui-sans-serif, sans-serif";
  const tw = ctx.measureText(text).width;
  canvas.width = Math.min(tw + 22, 260);
  canvas.height = 40;
  ctx.font = "bold 26px Inter, ui-sans-serif, sans-serif";
  ctx.fillStyle = "rgba(5, 8, 22, 0.82)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const hex = `#${color.toString(16).padStart(6, "0")}`;
  ctx.fillStyle = hex;
  ctx.fillText(text, 10, 29);
  return canvas;
}

function fibonacciSphere(n, radius) {
  const points = [];
  const golden = Math.PI * (1 + Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    points.push(
      new THREE.Vector3(
        r * Math.cos(theta) * radius,
        y * radius * 0.75,
        r * Math.sin(theta) * radius,
      ),
    );
  }
  return points;
}

export default function Visualizer({ text }) {
  const containerRef = useRef(null);

  const tokens = useMemo(
    () => text.trim().split(/\s+/).filter(Boolean).slice(0, 64),
    [text],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const W = container.clientWidth;
    const H = container.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050816, 0.032);

    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 200);
    camera.position.set(0, 4, 24);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xb4c6ff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(10, 14, 8);
    scene.add(sun);
    const fill = new THREE.PointLight(0x59e3ff, 1.4, 70);
    fill.position.set(-12, -6, -10);
    scene.add(fill);

    const n = Math.max(tokens.length, 8);
    const positions = fibonacciSphere(n, 9);

    // sequential edges
    const seqMat = new THREE.LineBasicMaterial({ color: 0x2a5070, transparent: true, opacity: 0.5 });
    for (let i = 0; i < n - 1; i++) {
      const geo = new THREE.BufferGeometry().setFromPoints([positions[i], positions[i + 1]]);
      scene.add(new THREE.Line(geo, seqMat));
    }

    // cross-edges: same first-3-chars tokens (vocabulary clustering)
    const crossMat = new THREE.LineBasicMaterial({ color: 0x7cf9ff, transparent: true, opacity: 0.22 });
    for (let i = 0; i < tokens.length; i++) {
      for (let j = i + 2; j < tokens.length; j++) {
        if (
          tokens[i].slice(0, 3).toLowerCase() === tokens[j].slice(0, 3).toLowerCase()
          && tokens[i].length > 2
        ) {
          const geo = new THREE.BufferGeometry().setFromPoints([positions[i], positions[j]]);
          scene.add(new THREE.Line(geo, crossMat));
          break;
        }
      }
    }

    // nodes + labels
    const nodeGroup = new THREE.Group();
    scene.add(nodeGroup);
    const nodeMeshes = [];

    for (let i = 0; i < n; i++) {
      const word = tokens[i] || `t${i}`;
      const color = nodeColor(word, i);
      const radius = 0.2 + Math.min(word.length, 12) * 0.016;

      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 22, 22),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.4,
          metalness: 0.55,
          roughness: 0.22,
        }),
      );
      mesh.position.copy(positions[i]);
      nodeGroup.add(mesh);
      nodeMeshes.push(mesh);

      const canvas = makeLabel(word, color);
      const tex = new THREE.CanvasTexture(canvas);
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.9, depthTest: false }),
      );
      sprite.scale.set(canvas.width / 52, canvas.height / 52, 1);
      sprite.position.copy(positions[i]);
      sprite.position.y += radius + 0.52;
      scene.add(sprite);
    }

    // grid
    const grid = new THREE.GridHelper(36, 22, 0x1a3a5c, 0x0c1e30);
    grid.position.y = -9;
    scene.add(grid);

    const onResize = () => {
      const nw = container.clientWidth;
      const nh = container.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    let frame = 0;
    let rafId;

    const animate = () => {
      frame++;
      const t = frame * 0.007;

      // orbit camera
      camera.position.x = Math.sin(t * 0.28) * 24;
      camera.position.z = Math.cos(t * 0.28) * 24;
      camera.position.y = 4 + Math.sin(t * 0.11) * 2.5;
      camera.lookAt(0, 0, 0);

      // node breathe
      nodeMeshes.forEach((mesh, i) => {
        const s = 1 + 0.06 * Math.sin(t * 1.8 + i * 0.65);
        mesh.scale.setScalar(s);
      });

      renderer.render(scene, camera);
      rafId = requestAnimationFrame(animate);
    };

    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      scene.traverse((o) => {
        o.geometry?.dispose();
        if (o.material) {
          if (Array.isArray(o.material)) {
            o.material.forEach((m) => { m.map?.dispose(); m.dispose(); });
          } else {
            o.material.map?.dispose();
            o.material.dispose();
          }
        }
      });
      renderer.dispose();
      container.innerHTML = "";
    };
  }, [tokens]);

  return <div ref={containerRef} className="visualizer-shell" />;
}
