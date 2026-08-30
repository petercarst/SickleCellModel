<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="csrf-token" content="{{ csrf_token() }}">
  <title>@yield('title', 'SickleVision — Cell Detection')</title>

  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet" />

  {{-- Palette matches uhAI Intelligence's brand design tokens (assets/css/variables.css) --}}
  <style>
    :root {
      /* ---- uhAI Intelligence brand tokens ---- */
      --color-navy-900: #071a3d;
      --color-navy-800: #0b2452;
      --color-navy-700: #103068;
      --color-green-700: #158249;
      --color-green-600: #1a9650;
      --color-green-500: #20a65a;
      --color-sky-700: #1364a8;
      --color-sky-600: #1976d2;
      --color-sky-500: #1e88e5;
      --color-error: #e5484d;
      --color-gray-50: #f8faf9;
      --color-gray-200: #e2e7ef;

      --bg:      var(--color-gray-50);
      --surface: #ffffff;
      --border:  #e6eaf1;
      --accent:  var(--color-green-600);
      --accent2: var(--color-sky-600);
      --normal:  var(--color-green-700);
      --sickle:  var(--color-error);
      --text:    #12151f;
      --text2:   #4b5568;
      --muted:   #6b7789;

      --tint-green: rgba(32,166,90,.1);
      --tint-sky: #eaf4fe;
      --tint-sky-icon: var(--color-sky-600);
      --tint-navy: #eef1f7;
      --tint-navy-icon: var(--color-navy-700);
      --tint-error: rgba(229,72,77,.1);

      --radius-sm: 10px;
      --radius-md: 16px;
      --radius-lg: 24px;
      --radius-pill: 999px;
      --shadow-sm: 0 2px 10px -4px hsla(220,30%,20%,.1);
      --shadow-md: 0 10px 28px -8px hsla(220,30%,20%,.12);
      --shadow-lg: 0 22px 52px -16px hsla(220,30%,20%,.16);
      --shadow-glow-green: 0 12px 30px -10px rgba(26,150,80,.45);
      --shadow-glow-sky: 0 12px 30px -10px rgba(25,118,210,.4);

      --display: 'Plus Jakarta Sans', 'Inter', 'Segoe UI', sans-serif;
      --mono:    'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
      --sans:    'Inter', 'Segoe UI', sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      overflow-x: hidden;
    }

    .navbar-brand {
      font-family: var(--display);
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--color-navy-900) !important;
      letter-spacing: 0;
    }
    .navbar-brand span { color: var(--accent); }
    .navbar-brand i { color: var(--accent); }
    nav {
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,.82) !important;
      backdrop-filter: blur(12px);
    }
    .nav-links a {
      font-family: var(--mono);
      font-size: .78rem;
      letter-spacing: .05em;
      color: var(--muted);
      text-decoration: none;
      margin-left: 1.1rem;
      transition: color .2s;
    }
    .nav-links a:hover, .nav-links a.active { color: var(--accent2); }
    .nav-links a.active { color: var(--accent2); font-weight: 600; }

    .hero {
      position: relative;
      z-index: 1;
      padding: 3rem 1rem 2rem;
      text-align: center;
      overflow: hidden;
    }
    @media (min-width: 768px) { .hero { padding: 4rem 0 2.5rem; } }
    .hero::before {
      content: '';
      position: absolute;
      inset: 0;
      background-image: radial-gradient(rgba(7,26,61,.08) 1px, transparent 1.5px);
      background-size: 26px 26px;
      opacity: .6;
      pointer-events: none;
      z-index: -1;
    }
    .hero-label {
      font-family: var(--mono);
      font-size: .72rem;
      letter-spacing: .2em;
      text-transform: uppercase;
      color: var(--color-green-700);
      margin-bottom: .85rem;
      font-weight: 600;
    }
    .hero h1 {
      font-family: var(--display);
      font-size: clamp(1.55rem, 5vw, 3rem);
      font-weight: 700;
      line-height: 1.2;
      color: var(--color-navy-900);
    }
    .hero h1 em { font-style: normal; color: var(--accent2); }
    .hero p {
      color: var(--text2);
      font-size: .95rem;
      max-width: 500px;
      margin: 1rem auto 0;
      line-height: 1.7;
    }

    .app-card {
      position: relative;
      z-index: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      overflow: hidden;
      box-shadow: var(--shadow-sm);
      transition: box-shadow .3s, transform .3s;
    }
    .app-card:hover { box-shadow: var(--shadow-md); }

    #dropZone {
      border: 2px dashed var(--color-gray-200);
      border-radius: var(--radius-md);
      padding: 1.75rem 1rem;
      cursor: pointer;
      transition: all .25s ease;
      background: var(--bg);
      text-align: center;
    }
    @media (min-width: 768px) { #dropZone { padding: 2.5rem 1.5rem; } }
    #dropZone:hover, #dropZone.drag-over {
      border-color: var(--accent2);
      background: var(--tint-sky);
    }
    #dropZone .drop-icon {
      font-size: 2.4rem;
      color: var(--muted);
      display: block;
      margin-bottom: .6rem;
      transition: color .25s, transform .25s;
    }
    #dropZone:hover .drop-icon { color: var(--accent2); transform: scale(1.1); }
    #dropZone p { color: var(--muted); font-size: .88rem; margin: 0; }
    #dropZone strong { color: var(--text); }

    #preview {
      display: none;
      max-height: 200px;
      border-radius: var(--radius-sm);
      object-fit: contain;
      border: 1px solid var(--border);
      width: 100%;
      margin-top: .75rem;
    }

    .btn-scan {
      background: var(--accent);
      color: #fff;
      border: none;
      font-family: var(--display);
      font-weight: 600;
      font-size: .85rem;
      letter-spacing: .01em;
      padding: .75rem 1.25rem;
      border-radius: var(--radius-pill);
      transition: all .25s;
      flex: 1 1 auto;
      box-shadow: var(--shadow-glow-green);
    }
    .btn-scan:disabled { opacity: .4; cursor: not-allowed; box-shadow: none; }
    .btn-scan:not(:disabled):hover {
      background: var(--color-green-700);
      transform: translateY(-2px);
      box-shadow: 0 18px 40px -10px rgba(26,150,80,.5);
    }
    .btn-reset {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text2);
      font-family: var(--display);
      font-weight: 600;
      font-size: .85rem;
      padding: .65rem .9rem;
      border-radius: var(--radius-pill);
      transition: all .2s;
      flex: 0 0 auto;
    }
    .btn-reset:hover { border-color: var(--accent2); color: var(--accent2); background: var(--tint-sky); }

    .scan-ring {
      width: 52px; height: 52px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .result-badge {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      font-family: var(--mono);
      font-size: 1.1rem;
      font-weight: 600;
      padding: .5rem 1.1rem;
      border-radius: var(--radius-pill);
    }
    .result-badge.sickle {
      background: var(--tint-error);
      color: var(--sickle);
      border: 1px solid rgba(229,72,77,.25);
    }
    .result-badge.normal {
      background: var(--tint-green);
      color: var(--normal);
      border: 1px solid rgba(32,166,90,.22);
    }

    .conf-track {
      height: 7px;
      background: var(--color-gray-200);
      border-radius: var(--radius-pill);
      overflow: hidden;
    }
    .conf-fill {
      height: 100%;
      border-radius: var(--radius-pill);
      transition: width 1s cubic-bezier(.4,0,.2,1);
      width: 0%;
    }
    .conf-fill.sickle { background: linear-gradient(90deg, #c53030, var(--sickle)); }
    .conf-fill.normal { background: linear-gradient(90deg, var(--color-green-500), var(--color-green-600)); }

    .stat-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: .8rem 1rem;
    }
    .stat-label {
      font-family: var(--mono);
      font-size: .58rem;
      letter-spacing: .15em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: .2rem;
    }
    .stat-value {
      font-family: var(--mono);
      font-size: 1rem;
      font-weight: 600;
      color: var(--color-navy-900);
      word-break: break-all;
    }

    .error-box {
      background: var(--tint-error);
      border: 1px solid rgba(229,72,77,.25);
      border-radius: var(--radius-md);
      padding: 1rem 1.1rem;
      color: #c53030;
      font-family: var(--mono);
      font-size: .8rem;
    }

    .section-label {
      font-family: var(--mono);
      font-size: .6rem;
      letter-spacing: .2em;
      text-transform: uppercase;
      color: var(--muted);
      padding-bottom: .5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1rem;
    }

    .info-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 1.2rem;
      height: 100%;
      box-shadow: var(--shadow-sm);
    }
    .info-card .icon {
      width: 38px; height: 38px;
      border-radius: var(--radius-sm);
      display: flex; align-items: center; justify-content: center;
      font-size: 1.1rem;
      margin-bottom: .8rem;
    }
    .info-card h6 {
      font-family: var(--display);
      font-weight: 700;
      font-size: .85rem;
      color: var(--color-navy-900);
      margin-bottom: .35rem;
    }
    .info-card p { font-size: .83rem; color: var(--text2); line-height: 1.6; margin: 0; }

    /* Stats strip (MySQL-backed) */
    .stats-strip {
      position: relative; z-index: 1;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: .75rem;
      margin-bottom: 1.5rem;
    }
    @media (min-width: 768px) { .stats-strip { grid-template-columns: repeat(4, 1fr); } }
    .stats-strip .stat-box { text-align: center; padding: 1rem .75rem; background: var(--surface); box-shadow: var(--shadow-sm); }
    .stats-strip .stat-value { font-size: 1.3rem; color: var(--color-navy-900); }

    table.history-table {
      font-family: var(--sans);
      color: var(--text);
      width: 100%;
    }
    table.history-table th {
      font-family: var(--mono);
      font-size: .62rem;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      padding: .6rem .5rem;
      text-align: left;
    }
    table.history-table td {
      padding: .65rem .5rem;
      border-bottom: 1px solid var(--border);
      font-size: .85rem;
    }
    table.history-table tr:last-child td { border-bottom: none; }

    footer {
      border-top: 1px solid var(--border);
      font-family: var(--mono);
      font-size: .7rem;
      color: var(--muted);
      padding: 1.2rem 1rem;
      text-align: center;
      position: relative; z-index: 1;
      background: var(--surface);
    }

    .fade-in { animation: fadeIn .45s ease forwards; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

    .pulse-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--color-green-500);
      box-shadow: 0 0 8px rgba(32,166,90,.7);
      animation: pulse 2s ease-in-out infinite;
      margin-right: .35rem;
    }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

    .disclaimer {
      background: var(--tint-sky);
      border: 1px solid rgba(25,118,210,.18);
      border-radius: var(--radius-md);
      padding: .85rem 1rem;
      font-size: .78rem;
      color: var(--color-sky-700);
    }

    .result-inner {
      min-height: 240px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    @media (min-width: 768px) { .result-inner { min-height: 340px; } }

    @yield('extra-style')
  </style>
</head>
<body>

<nav class="navbar navbar-light sticky-top px-3">
  <span class="navbar-brand">
    <i class="bi bi-activity me-2"></i>Sickle<span>Vision</span>
  </span>
  <div class="d-flex align-items-center">
    <span class="nav-links d-none d-sm-inline-block">
      <a href="{{ route('predictions.index') }}" class="{{ request()->routeIs('predictions.index') ? 'active' : '' }}">Scan</a>
      <a href="{{ route('predictions.history') }}" class="{{ request()->routeIs('predictions.history') ? 'active' : '' }}">History</a>
    </span>
    <span style="font-family:var(--mono);font-size:.66rem;color:var(--muted);margin-left:1.1rem;">
      <span class="pulse-dot"></span>MODEL ONLINE
    </span>
  </div>
</nav>

@yield('content')

<footer>
  <div class="container">
    SickleVision &mdash; AI Hematology Tool &nbsp;·&nbsp;
    Laravel + MySQL + TensorFlow Serving &nbsp;·&nbsp;
    <span style="color:var(--accent);">For Research Use Only</span>
    <br>
    <span style="font-style: italic; color: var(--color-sky-700);">Developed by Peter Carst.</span>
    <span style="font-style: italic; color: var(--color-green-700);">Powered by uhAi_930</span>
  </div>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
@yield('scripts')
</body>
</html>
