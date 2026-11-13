"""Styling and theme definitions for the Research Studio.

Follows the Stage 11 design direction:
- Premium dark research laboratory aesthetic
- Deep navy / charcoal background
- Restrained cyan and electric blue accents
- Status-coded indicators (emerald green for verified, amber for warning, red for error)
- Professional typography, high contrast, compact metric cards
"""

STUDIO_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, pre, [data-testid="stMarkdownContainer"] code {
        font-family: 'JetBrains Mono', Consolas, Monaco, monospace !important;
        font-size: 0.88rem !important;
    }

    /* Studio Header */
    .studio-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #132238 50%, #0a111c 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .studio-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .studio-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Verification Badges */
    .badge-verified {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-warning {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-unverified {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Metric Cards */
    .metric-card-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: #111c2a;
        border: 1px solid #1e324d;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 4px 0;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card-container:hover {
        border-color: #38bdf8;
    }
    .card-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .card-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .card-sub {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* Callout & Technical Note Boxes */
    .tech-note-box {
        background-color: #0c1622;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .tech-alert-amber {
        background-color: #1a160c;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        color: #fed7aa;
        font-size: 0.92rem;
    }
    .tech-alert-emerald {
        background-color: #0b1a13;
        border-left: 4px solid #10b981;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        color: #a7f3d0;
        font-size: 0.92rem;
    }

    /* Section Subheadings */
    .lab-section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        border-left: 3px solid #38bdf8;
        padding-left: 10px;
        margin: 24px 0 12px 0;
        letter-spacing: -0.01em;
    }

    /* Source file tag */
    .source-tag {
        display: inline-block;
        background-color: #1e293b;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #334155;
        margin-top: 4px;
    }
</style>
"""
