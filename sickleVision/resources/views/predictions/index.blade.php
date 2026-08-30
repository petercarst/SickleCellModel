@extends('layouts.app')

@section('title', 'SickleVision — Cell Detection')

@section('content')

<section class="hero">
  <div class="container">
    <p class="hero-label">AI-Powered Hematology</p>
    <h1>Detect <em>Sickle Cell</em><br>from Blood Smear Images</h1>
    <p>Upload a microscopy image of red blood cells. The deep learning model classifies them as sickle cell or normal in seconds.</p>
  </div>
</section>

<main class="container pb-5" style="position:relative;z-index:1;">

  <div class="stats-strip">
    <div class="stat-box">
      <div class="stat-label">Total Scans</div>
      <div class="stat-value" id="statTotalScans">{{ number_format($stats['total']) }}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Sickle-Positive</div>
      <div class="stat-value" id="statSicklePositive">{{ number_format($stats['sickle']) }}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Positive Rate</div>
      <div class="stat-value" id="statPositiveRate">{{ $stats['sickle_pct'] }}%</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Avg. Confidence</div>
      <div class="stat-value" id="statAvgConfidence">{{ $stats['avg_confidence'] }}%</div>
    </div>
  </div>

  <div class="row g-3 g-md-4">

    <div class="col-12 col-md-6">
      <div class="app-card p-3 p-md-4 d-flex flex-column" style="gap:.9rem;">
        <div class="section-label">01 — Upload Image</div>

        <div id="dropZone" onclick="document.getElementById('fileInput').click()">
          <i class="bi bi-cloud-upload drop-icon"></i>
          <p><strong>Drop your image here</strong></p>
          <p class="mt-1" style="font-size:.76rem;">
            or click to browse &nbsp;·&nbsp; JPEG · JPG · PNG · WebP &nbsp;·&nbsp; max 10 MB
          </p>
          <img id="preview" src="" alt="Preview" />
        </div>

        <input type="file" id="fileInput"
               accept="image/jpeg,image/jpg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
               class="d-none" />

        <div id="fileInfo" class="stat-box" style="display:none;">
          <div class="stat-label">Selected File</div>
          <div id="fileName" class="stat-value" style="font-size:.82rem;"></div>
        </div>

        <div class="d-flex gap-2">
          <button class="btn-scan" id="scanBtn" disabled onclick="runPrediction()">
            <i class="bi bi-cpu me-1"></i>RUN SCAN
          </button>
          <button class="btn-reset" id="resetBtn" onclick="resetAll()" title="Reset">
            <i class="bi bi-arrow-counterclockwise"></i>
          </button>
        </div>

        <div class="disclaimer">
          <i class="bi bi-info-circle me-1"></i>
          <strong>For research use only.</strong> Not a substitute for professional medical diagnosis.
        </div>
      </div>
    </div>

    <div class="col-12 col-md-6">
      <div class="app-card p-3 p-md-4 d-flex flex-column">
        <div class="section-label">02 — Analysis Result</div>

        <div class="result-inner">

          <div id="idleState" class="d-flex flex-column align-items-center justify-content-center text-center" style="gap:.7rem;">
            <i class="bi bi-eyedropper" style="font-size:2.6rem;color:var(--border);"></i>
            <p style="color:var(--muted);font-family:var(--mono);font-size:.72rem;margin:0;">
              Awaiting image upload<br>and scan execution
            </p>
          </div>

          <div id="loadingState" class="d-none flex-column align-items-center justify-content-center text-center" style="gap:.8rem;">
            <div class="scan-ring"></div>
            <p style="font-family:var(--mono);font-size:.75rem;color:var(--muted);margin:0;">Analyzing blood cells…</p>
          </div>

          <div class="error-box d-none" id="errorBox">
            <i class="bi bi-exclamation-triangle me-2"></i>
            <span id="errorMsg"></span>
          </div>

          <div id="resultBox" class="d-none">
            <div class="text-center mb-3">
              <div id="resultBadge" class="result-badge mb-2"></div>
              <p id="resultDesc" style="color:var(--muted);font-size:.83rem;max-width:300px;margin:0 auto;line-height:1.6;"></p>
            </div>

            <div class="mb-3">
              <div class="d-flex justify-content-between mb-1">
                <span style="font-family:var(--mono);font-size:.65rem;color:var(--muted);">CONFIDENCE</span>
                <span id="confPct" style="font-family:var(--mono);font-size:.65rem;color:var(--text);"></span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" id="confFill"></div>
              </div>
            </div>

            <div class="row g-2">
              <div class="col-6">
                <div class="stat-box">
                  <div class="stat-label">Prediction</div>
                  <div class="stat-value" id="statClass">—</div>
                </div>
              </div>
              <div class="col-6">
                <div class="stat-box">
                  <div class="stat-label">Confidence</div>
                  <div class="stat-value" id="statScore">—</div>
                </div>
              </div>
              <div class="col-12">
                <div class="stat-box">
                  <div class="stat-label">Raw Sigmoid Output</div>
                  <div class="stat-value" id="statRaw">—</div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

  </div>

  <div class="row mt-4">
    <div class="col-12">
      <div class="app-card p-3 p-md-4">
        <div class="d-flex justify-content-between align-items-center section-label" style="border-bottom:1px solid var(--border);">
          <span>03 — Recent Scans</span>
          <a href="{{ route('predictions.history') }}" style="font-family:var(--mono);font-size:.65rem;color:var(--accent2);text-decoration:none;">View all &rarr;</a>
        </div>
        <div id="recentScansEmpty" style="padding:1rem 0;text-align:center;color:var(--muted);font-family:var(--mono);font-size:.75rem;{{ $recent->isNotEmpty() ? 'display:none;' : '' }}">
          No scans yet — run one above.
        </div>
        <div id="recentScansWrap" style="overflow-x:auto;{{ $recent->isEmpty() ? 'display:none;' : '' }}">
          <table class="history-table">
            <thead>
              <tr><th>File</th><th>Result</th><th>Confidence</th><th>When</th></tr>
            </thead>
            <tbody id="recentScansBody">
              @foreach ($recent as $p)
              <tr>
                <td>{{ \Illuminate\Support\Str::limit($p->original_filename, 28) }}</td>
                <td><span style="color: {{ $p->isSickleCell() ? 'var(--sickle)' : 'var(--normal)' }};">{{ $p->predicted_class }}</span></td>
                <td>{{ number_format($p->confidence * 100, 1) }}%</td>
                <td style="color:var(--muted);">{{ $p->created_at->diffForHumans() }}</td>
              </tr>
              @endforeach
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3 mt-2">
    <div class="col-12 col-sm-6 col-md-4">
      <div class="info-card">
        <div class="icon" style="background:var(--tint-navy);color:var(--tint-navy-icon);">
          <i class="bi bi-image"></i>
        </div>
        <h6>Image Requirements</h6>
        <p>Upload Giemsa-stained peripheral blood smear microscopy images in JPEG, JPG, PNG, or WebP format. Higher resolution yields better accuracy.</p>
      </div>
    </div>
    <div class="col-12 col-sm-6 col-md-4">
      <div class="info-card">
        <div class="icon" style="background:var(--tint-sky);color:var(--tint-sky-icon);">
          <i class="bi bi-cpu"></i>
        </div>
        <h6>How It Works</h6>
        <p>Laravel forwards your image to a Python TensorFlow API, which preprocesses it and queries a CNN served via TensorFlow Serving. Results are logged to MySQL and returned in real time.</p>
      </div>
    </div>
    <div class="col-12 col-sm-6 col-md-4">
      <div class="info-card">
        <div class="icon" style="background:var(--tint-green);color:var(--normal);">
          <i class="bi bi-shield-check"></i>
        </div>
        <h6>Privacy</h6>
        <p>Images themselves are never stored — only the prediction result (class, confidence, filename, timestamp) is logged to the database.</p>
      </div>
    </div>
  </div>

</main>

@endsection

@section('scripts')
<script>
  const dropZone    = document.getElementById('dropZone');
  const fileInput   = document.getElementById('fileInput');
  const preview     = document.getElementById('preview');
  const fileInfo    = document.getElementById('fileInfo');
  const fileName    = document.getElementById('fileName');
  const scanBtn     = document.getElementById('scanBtn');
  const idleState   = document.getElementById('idleState');
  const loadingState= document.getElementById('loadingState');
  const errorBox    = document.getElementById('errorBox');
  const resultBox   = document.getElementById('resultBox');
  const csrfToken   = document.querySelector('meta[name="csrf-token"]').content;

  let selectedFile = null;

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    const allowedMime = ['image/jpeg','image/jpg','image/png','image/webp'];
    const allowedExt  = ['jpg','jpeg','png','webp'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!allowedMime.includes(file.type) && !allowedExt.includes(ext)) {
      showError('Invalid file type. Please upload a JPEG, JPG, PNG, or WebP image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError('File too large. Maximum size is 10 MB.');
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      preview.src = ev.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    fileName.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.style.display = 'block';
    scanBtn.disabled = false;
    setPanel('blank');
  }

  async function runPrediction() {
    if (!selectedFile) return;
    scanBtn.disabled = true;
    setPanel('loading');

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
      const res = await fetch('{{ route('predictions.store') }}', {
          method: 'POST',
          headers: { 'X-CSRF-TOKEN': csrfToken, 'Accept': 'application/json' },
          body: fd
       });
      const data = await res.json();
      if (!res.ok || data.error) {
        showError(data.error || 'Something went wrong.');
        return;
      }
      renderResult(data);
    } catch {
      showError('Network error: could not reach the server. Please check your connection.');
    } finally {
      scanBtn.disabled = false;
    }
  }

  function renderResult(data) {
    const cls      = data.class;
    const conf     = data.confidence;
    const isSickle = cls === 'Sickle Cell';
    const pct      = (conf * 100).toFixed(1);
    const key      = isSickle ? 'sickle' : 'normal';

    const badge = document.getElementById('resultBadge');
    badge.className = `result-badge ${key}`;
    badge.innerHTML = `<i class="bi bi-${isSickle ? 'exclamation-circle' : 'check-circle'}"></i> ${cls}`;

    document.getElementById('resultDesc').textContent = isSickle
      ? 'The model identified characteristics consistent with sickle-shaped red blood cells.'
      : 'The model did not detect significant sickle cell morphology in this sample.';

    const fill = document.getElementById('confFill');
    fill.className = `conf-fill ${key}`;
    fill.style.width = '0%';
    document.getElementById('confPct').textContent = `${pct}%`;
    setTimeout(() => { fill.style.width = `${pct}%`; }, 60);

    document.getElementById('statClass').textContent = cls;
    document.getElementById('statScore').textContent = `${pct}%`;
    document.getElementById('statRaw').textContent   = Number(data.raw_score ?? conf).toFixed(6);

    if (data.stats) updateStats(data.stats);
    if (selectedFile) prependRecentScan(selectedFile.name, cls, conf);

    setPanel('result');
  }

  function updateStats(stats) {
    document.getElementById('statTotalScans').textContent    = stats.total.toLocaleString();
    document.getElementById('statSicklePositive').textContent = stats.sickle.toLocaleString();
    document.getElementById('statPositiveRate').textContent   = `${stats.sickle_pct}%`;
    document.getElementById('statAvgConfidence').textContent  = `${stats.avg_confidence}%`;
  }

  function truncateName(name, len = 28) {
    return name.length > len ? name.slice(0, len) + '...' : name;
  }

  function prependRecentScan(filename, cls, confidence) {
    const tbody = document.getElementById('recentScansBody');
    if (!tbody) return;

    const tr = document.createElement('tr');

    const fileTd = document.createElement('td');
    fileTd.textContent = truncateName(filename);

    const resultTd = document.createElement('td');
    const span = document.createElement('span');
    span.style.color = cls === 'Sickle Cell' ? 'var(--sickle)' : 'var(--normal)';
    span.textContent = cls;
    resultTd.appendChild(span);

    const confTd = document.createElement('td');
    confTd.textContent = `${(confidence * 100).toFixed(1)}%`;

    const whenTd = document.createElement('td');
    whenTd.style.color = 'var(--muted)';
    whenTd.textContent = 'just now';

    tr.append(fileTd, resultTd, confTd, whenTd);
    tbody.prepend(tr);
    while (tbody.children.length > 5) tbody.removeChild(tbody.lastElementChild);

    document.getElementById('recentScansEmpty').style.display = 'none';
    document.getElementById('recentScansWrap').style.display = '';
  }

  function showError(msg) {
    setPanel('error');
    document.getElementById('errorMsg').textContent = msg;
  }

  function setPanel(state) {
    idleState.classList.add('d-none');    idleState.classList.remove('d-flex');
    loadingState.classList.add('d-none'); loadingState.classList.remove('d-flex');
    errorBox.classList.add('d-none');
    resultBox.classList.add('d-none');

    switch (state) {
      case 'idle':
        idleState.classList.remove('d-none'); idleState.classList.add('d-flex'); break;
      case 'loading':
        loadingState.classList.remove('d-none'); loadingState.classList.add('d-flex'); break;
      case 'error':
        errorBox.classList.remove('d-none'); break;
      case 'result':
        resultBox.classList.remove('d-none');
        resultBox.classList.remove('fade-in');
        void resultBox.offsetWidth;
        resultBox.classList.add('fade-in');
        break;
      case 'blank':
      default:
        break;
    }
  }

  function resetAll() {
    selectedFile          = null;
    fileInput.value       = '';
    preview.src           = '';
    preview.style.display = 'none';
    fileInfo.style.display= 'none';
    scanBtn.disabled      = true;
    document.getElementById('confFill').style.width = '0%';
    setPanel('idle');
  }

  setPanel('idle');
</script>
@endsection
