    if (qs.has("sub") && home.gaitOn) placeCat(home.t, +qs.get("sub"));  // local checks only: a 1/30 s step
    if (qs.has("facecam")) {  // local checks only: camera beside the cat at its current pose, turned by facecam degrees
      home.world.updateMatrixWorld(true);
      const M = new THREE.Matrix4().multiplyMatrices(home.world.matrixWorld, home.catHolder.matrix);
      const f = new THREE.Vector3(0, 0, +(qs.get("fsign") || -1)).transformDirection(M); f.y = 0; f.normalize();
      const c = new THREE.Vector3().setFromMatrixPosition(M);
      const o = home.view.pos.clone().sub(home.view.target), L = o.length() * (+qs.get("fdist") || 0.35), e = (+(qs.get("felev") || 20)) * Math.PI / 180;
      const f0 = f.clone(); c.add(f0.multiplyScalar(+(qs.get("fx") || 0))).add(new THREE.Vector3(0, +(qs.get("fy") || 0), 0));
      f.applyAxisAngle(new THREE.Vector3(0, 1, 0), +qs.get("facecam") * Math.PI / 180);
      home.view.target.copy(c); home.view.pos.copy(c).add(f.multiplyScalar(L * Math.cos(e))).add(new THREE.Vector3(0, L * Math.sin(e), 0));
      camera.position.copy(home.view.pos); controls.target.copy(home.view.target); $("home-follow").checked = false;
    }
    if (qs.has("follow")) $("home-follow").checked = true;  // local before/after only
    if (qs.has("speed")) $("home-speed").value = qs.get("speed");
    if (qs.has("play")) { home.playing = true; $("home-play").textContent = "Pause"; }
    if (qs.has("rec")) {  // local before/after only: render frames t0..t1 (3 steps each) from a side camera; room hidden
      // (grid floor) unless room=1; the contact shadows stay
      const [t0, t1] = qs.get("rec").split(",").map(Number), tag = qs.get("tag") || "x", W = 960, Hh = 540;
      renderer.setPixelRatio(1); renderer.setSize(W, Hh, false); camera.aspect = W / Hh; camera.updateProjectionMatrix();
      const keep = (o) => o === home.catHolder || (home.shadows && (o === home.shadows.body || home.shadows.paws.includes(o)));
      if (!qs.has("room")) {
        home.world.children.forEach((o) => { if (!keep(o)) o.visible = false; });
        const grid = new THREE.GridHelper(16, 160, 0x9a948c, 0x5a5650);
        grid.position.set(0, -home.floorY, 0);
        home.world.add(grid);
        home.scene.background = new THREE.Color("#2a2826");
      } else {
        [home.allPaths, home.active, home.person, home.markers].forEach((o) => { if (o) o.visible = false; });
      }
      if (home.skel) home.skel.visible = false;
      home.world.updateMatrixWorld(true);
      const tr = home.tracks[home.k], r = tr.root;
      const pA = new THREE.Vector3(r[3 * t0], r[3 * t0 + 1], r[3 * t0 + 2]).applyMatrix4(home.world.matrixWorld);
      const pB = new THREE.Vector3(r[3 * t1], r[3 * t1 + 1], r[3 * t1 + 2]).applyMatrix4(home.world.matrixWorld);
      const dir = pB.clone().sub(pA); dir.y = 0; dir.normalize();
      const side = new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(+(qs.get("flip") || 1));
      const D = +(qs.get("dist") || 1.2), Hc = +(qs.get("hgt") || 0.3);
      let tgt = null, k = 0;
      for (let t = t0; t <= t1; t++) for (let sb = 0; sb < 3; sb++) {
        placeCat(t, sb);
        home.world.updateMatrixWorld(true);
        const c = new THREE.Vector3().setFromMatrixPosition(new THREE.Matrix4().multiplyMatrices(home.world.matrixWorld, home.catHolder.matrix));
        tgt = tgt ? tgt.lerp(c, 0.12) : c.clone();
        camera.position.copy(tgt).addScaledVector(side, D).add(new THREE.Vector3(0, Hc, 0));
        camera.lookAt(tgt.x, tgt.y - 0.05, tgt.z);
        renderer.setSize(W, Hh, false); camera.aspect = W / Hh; camera.updateProjectionMatrix();  // a resize may have changed them
        renderer.render(home.scene, camera);
        const blob = await new Promise((ok) => renderer.domElement.toBlob(ok, "image/jpeg", 0.9));
        await fetch("/rec/" + tag + "_" + String(k).padStart(4, "0") + ".jpg", { method: "POST", body: blob });
        k++;
      }
      document.body.dataset.rec = String(k);
    }
    if (qs.has("shadowtime")) {  // local checks only: what the contact shadows add per pose
      const k0 = home.k, ti = (on) => { home.shadows.off = !on; const t0 = performance.now();
        for (let i = 0; i < 60; i++) placeCat(100 + (i % 30), 0); return (performance.now() - t0) / 60; };
      const off = ti(false), on = ti(true);
      home.shadows.off = false; setVideo(k0);
      document.body.dataset.shadowtime = JSON.stringify({ ms_per_pose_without: +off.toFixed(2), ms_per_pose_with: +on.toFixed(2),
        paw_vertices: home.a2s.paws.map((s) => s.length) });
    }
    if (qs.has("shadowdbg")) {  // local checks only: where the shadows are, against the room scan under the cat
      const S = home.shadows, b = S.body.position, rp = home.world.children[0].geometry.attributes.position.array;
      let near = [];
      for (let i = 0; i < rp.length; i += 3) if (Math.hypot(rp[i] - b.x, rp[i + 2] - b.z) < 0.15) near.push(rp[i + 1]);
      near.sort((x, y) => x - y);
      const q = (f) => near.length ? +near[Math.floor(f * (near.length - 1))].toFixed(3) : null;
      document.body.dataset.sh = JSON.stringify({ floor: +(-home.floorY).toFixed(3), body: [+b.x.toFixed(3), +b.y.toFixed(3), +b.z.toFixed(3)],
        visible: S.body.visible, opacity: +S.body.material.opacity.toFixed(2), scale: [+S.body.scale.x.toFixed(3), +S.body.scale.z.toFixed(3)],
        paws: S.paws.map((m) => [+m.position.y.toFixed(3), +m.material.opacity.toFixed(2), m.visible]),
        scan_y_within_15cm_q10_q50_q90: [q(0.1), q(0.5), q(0.9)], scan_points: near.length });
    }
    if (qs.has("shadowred")) {  // local checks only: paint the body shadow solid red to see where it is drawn
      const m = home.shadows.body.material; m.map = null; m.color.set(0xff0000); m.opacity = 1; m.transparent = false; m.needsUpdate = true;
      home.shadows.paws.forEach((p) => { p.material.map = null; p.material.color.set(0x0000ff); p.material.transparent = false; p.material.needsUpdate = true; });
    }
    if (qs.has("rendertime")) {  // local checks only: what the cast shadow adds to drawing a frame
      const gl = renderer.getContext(), px = new Uint8Array(4);
      const ti = (on) => { home.shadows.off = !on; placeCat(home.t, 0); renderer.render(home.scene, camera); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const t0 = performance.now(); for (let i = 0; i < 30; i++) { renderer.render(home.scene, camera); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); }
        return (performance.now() - t0) / 30; };
      const off = ti(false), on = ti(true), off2 = ti(false), on2 = ti(true);
      home.shadows.off = false; placeCat(home.t, 0);
      document.body.dataset.rendertime = JSON.stringify({ ms_per_frame_without: [+off.toFixed(2), +off2.toFixed(2)], ms_per_frame_with: [+on.toFixed(2), +on2.toFixed(2)] });
    }
