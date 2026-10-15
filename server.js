const express = require('express');
const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const sharp = require('sharp');

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff']);

let classifierPromise = null;
async function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = (async () => {
      const { pipeline } = await import('@xenova/transformers');
      // Pipeline de clasificacion zero-shot basada en CLIP (local, sin APIs)
      return await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
    })();
  }
  return classifierPromise;
}

function sanitizeFolderName(name) {
  return String(name)
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, ' ')
    .slice(0, 60) || 'criterio';
}

function uniquePath(dir, base) {
  let candidate = path.join(dir, base);
  if (!fs.existsSync(candidate)) return candidate;
  let i = 2;
  while (fs.existsSync(path.join(dir, `${base} (${i})`))) i++;
  return path.join(dir, `${base} (${i})`);
}

async function listImages(folder) {
  const entries = await fsp.readdir(folder, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && IMAGE_EXT.has(path.extname(e.name).toLowerCase()))
    .map((e) => path.join(folder, e.name));
}

// POST /api/classify  { folder, criteria: string[], threshold?, mode? }
app.post('/api/classify', async (req, res) => {
  const { folder, criteria = [], threshold = 0.2, mode = 'best' } = req.body || {};
  if (!folder || !Array.isArray(criteria) || criteria.length === 0) {
    return res.status(400).json({ error: 'folder y criteria son obligatorios' });
  }
  const absFolder = path.resolve(folder);
  if (!fs.existsSync(absFolder) || !fs.statSync(absFolder).isDirectory()) {
    return res.status(400).json({ error: 'La carpeta no existe: ' + absFolder });
  }

  const crits = criteria.map((c) => String(c).trim()).filter(Boolean);
  if (crits.length === 0) return res.status(400).json({ error: 'Sin criterios validos' });

  // SSE para progreso
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  try {
    const classifier = await getClassifier();
    const images = await listImages(absFolder);
    send({ type: 'start', total: images.length, criteria: crits });

    // preparar subcarpetas (no mover aun, solo nombres)
    const folderNames = crits.map(sanitizeFolderName);
    // make unique folder names
    const seen = new Set();
    const uniqueFolderNames = folderNames.map((n) => {
      if (!seen.has(n)) { seen.add(n); return n; }
      let i = 2; while (seen.has(`${n} (${i})`)) i++;
      const r = `${n} (${i})`; seen.add(r); return r;
    });

    const results = []; // { image, best, scores, targetFolder? }
    for (let i = 0; i < images.length; i++) {
      const imgPath = images[i];
      let scores = [];
      try {
        const out = await classifier(imgPath, crits);
        // out: array of {label, score}
        scores = crits.map((c) => {
          const found = out.find((r) => r.label === c);
          return { criterion: c, score: found ? found.score : 0 };
        });
      } catch (e) {
        scores = crits.map((c) => ({ criterion: c, score: 0 }));
      }

      // elegir mejor
      let best = null;
      let bestScore = -1;
      for (const s of scores) {
        if (s.score > bestScore) { bestScore = s.score; best = s.criterion; }
      }

      let target = null;
      if (bestScore >= threshold) {
        if (mode === 'all') {
          // mover a cada criterio que supera threshold
          target = scores.filter((s) => s.score >= threshold).map((s) => s.criterion);
        } else {
          target = [best];
        }
      } else {
        target = []; // sin clasificar
      }

      const targetFolders = target.map((c) => uniqueFolderNames[crits.indexOf(c)]);

      results.push({
        image: path.basename(imgPath),
        fullPath: imgPath,
        scores: scores.map((s) => ({ criterion: s.criterion, score: Number(s.score.toFixed(4)) })),
        best,
        bestScore: Number(bestScore.toFixed(4)),
        targetFolders,
      });

      send({ type: 'progress', current: i + 1, total: images.length, file: path.basename(imgPath), best, bestScore: Number(bestScore.toFixed(4)) });
    }

    send({ type: 'done', results, criteria: crits, folderNames: uniqueFolderNames, baseFolder: absFolder, threshold, mode });
    res.end();
  } catch (e) {
    send({ type: 'error', message: String(e && e.message ? e.message : e) });
    res.end();
  }
});

// POST /api/apply  { baseFolder, results: [{fullPath, targetFolders}], criteria, folderNames }
app.post('/api/apply', async (req, res) => {
  const { baseFolder, results = [], folderNames = [] } = req.body || {};
  if (!baseFolder || !Array.isArray(results)) {
    return res.status(400).json({ error: 'datos invalidos' });
  }
  const absBase = path.resolve(baseFolder);
  const created = new Set();
  const moved = [];
  const errors = [];

  for (const r of results) {
    if (!r.targetFolders || r.targetFolders.length === 0) continue;
    for (const fn of r.targetFolders) {
      const destDir = path.join(absBase, fn);
      if (!created.has(destDir)) {
        if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });
        created.add(destDir);
      }
      try {
        const dest = path.join(destDir, path.basename(r.fullPath));
        // evitar sobrescribir
        let finalDest = dest;
        if (fs.existsSync(finalDest)) {
          const ext = path.extname(dest);
          const base = path.basename(dest, ext);
          let i = 2;
          while (fs.existsSync(path.join(destDir, `${base} (${i})${ext}`))) i++;
          finalDest = path.join(destDir, `${base} (${i})${ext}`);
        }
        await fsp.rename(r.fullPath, finalDest);
        moved.push({ from: r.fullPath, to: finalDest });
      } catch (e) {
        errors.push({ file: r.fullPath, folder: fn, error: String(e.message || e) });
      }
    }
  }

  res.json({ movedCount: moved.length, errors, createdFolders: Array.from(created) });
});

// Listar contenido de una carpeta (para verificacion)
app.post('/api/list', async (req, res) => {
  const { folder } = req.body || {};
  if (!folder) return res.status(400).json({ error: 'folder requerido' });
  const abs = path.resolve(folder);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
    return res.status(400).json({ error: 'no existe' });
  }
  const entries = await fsp.readdir(abs, { withFileTypes: true });
  res.json({ folder: abs, entries: entries.map((e) => ({ name: e.name, dir: e.isDirectory(), file: e.isFile() })) });
});

// Utilidad: listar imagenes (solo nombres + rutas)
app.post('/api/images', async (req, res) => {
  const { folder } = req.body || {};
  if (!folder) return res.status(400).json({ error: 'folder requerido' });
  const abs = path.resolve(folder);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
    return res.status(400).json({ error: 'no existe' });
  }
  try {
    const files = await listImages(abs);
    const subs = (await fsp.readdir(abs, { withFileTypes: true }))
      .filter((e) => e.isDirectory()).map((e) => e.name);
    res.json({
      folder: abs,
      images: files.map((f) => ({ name: path.basename(f), fullPath: f })),
      subfolders: subs,
    });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

// Mover una sola imagen a una subcarpeta (crea la subcarpeta si no existe)
app.post('/api/move-one', async (req, res) => {
  const { fullPath, destName } = req.body || {};
  if (!fullPath || !destName) return res.status(400).json({ error: 'fullPath y destName requeridos' });
  const absFile = path.resolve(fullPath);
  if (!fs.existsSync(absFile) || !fs.statSync(absFile).isFile()) {
    return res.status(400).json({ error: 'archivo no existe' });
  }
  const baseDir = path.dirname(absFile);
  const folderName = sanitizeFolderName(destName);
  const destDir = path.join(baseDir, folderName);
  if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });
  try {
    const ext = path.extname(absFile);
    const base = path.basename(absFile, ext);
    let finalDest = path.join(destDir, path.basename(absFile));
    if (fs.existsSync(finalDest)) {
      let i = 2;
      while (fs.existsSync(path.join(destDir, `${base} (${i})${ext}`))) i++;
      finalDest = path.join(destDir, `${base} (${i})${ext}`);
    }
    await fsp.rename(absFile, finalDest);
    res.json({ ok: true, from: absFile, to: finalDest, folder: folderName });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

// Servir una imagen local para preview
app.get('/api/file', (req, res) => {
  const p = req.query.path;
  if (!p) return res.status(400).send('path requerido');
  const abs = path.resolve(String(p));
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) {
    return res.status(404).send('no existe');
  }
  res.sendFile(abs);
});

// Hash perceptivo (dHash) para detectar imagenes similares
async function dhash(filePath, size = 16) {
  const { data } = await sharp(filePath)
    .resize(size + 1, size, { fit: 'fill' })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  let bits = '';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * (size + 1) + x;
      bits += data[i] > data[i + 1] ? '1' : '0';
    }
  }
  return bits;
}
function hamming(a, b) {
  if (!a || !b || a.length !== b.length) return 999;
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

// Union-Find helper
function makeUF(n) {
  const p = Array.from({ length: n }, (_, i) => i);
  function find(x) { while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  function union(a, b) { const ra = find(a), rb = find(b); if (ra !== rb) p[ra] = rb; }
  return { find, union };
}

// Detectar grupos de imagenes duplicadas/similares
app.post('/api/duplicates', async (req, res) => {
  const { folder, threshold = 10 } = req.body || {};
  if (!folder) return res.status(400).json({ error: 'folder requerido' });
  const abs = path.resolve(folder);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
    return res.status(400).json({ error: 'no existe' });
  }
  try {
    const files = await listImages(abs);
    const items = [];
    for (const f of files) {
      try {
        const h = await dhash(f);
        items.push({ name: path.basename(f), fullPath: f, hash: h });
      } catch (e) {
        // ignorar archivos problematicos
      }
    }
    const { find, union } = makeUF(items.length);
    const thr = Math.max(0, Math.min(64, parseInt(threshold, 10) || 10));
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        if (hamming(items[i].hash, items[j].hash) <= thr) union(i, j);
      }
    }
    const groupsMap = new Map();
    for (let i = 0; i < items.length; i++) {
      const r = find(i);
      if (!groupsMap.has(r)) groupsMap.set(r, []);
      groupsMap.get(r).push(items[i]);
    }
    const groups = [];
    let gid = 1;
    for (const arr of groupsMap.values()) {
      if (arr.length >= 2) groups.push({ id: gid++, images: arr.map(({ name, fullPath }) => ({ name, fullPath })) });
    }
    res.json({ folder: abs, total: items.length, groupCount: groups.length, groups });
  } catch (e) {
    res.status(500).json({ error: String(e && e.message ? e.message : e) });
  }
});

// Eliminar un archivo
app.post('/api/delete-one', async (req, res) => {
  const { fullPath } = req.body || {};
  if (!fullPath) return res.status(400).json({ error: 'fullPath requerido' });
  const abs = path.resolve(fullPath);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) {
    return res.status(400).json({ error: 'archivo no existe' });
  }
  try {
    await fsp.unlink(abs);
    res.json({ ok: true, fullPath: abs });
  } catch (e) {
    res.status(500).json({ error: String(e && e.message ? e.message : e) });
  }
});

app.get('/api/health', (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Albumer corriendo en http://localhost:${PORT}`);
});
