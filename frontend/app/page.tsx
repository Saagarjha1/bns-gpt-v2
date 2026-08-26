import { API_BASE_URL } from "@/lib/api";

type HomePageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

function readError(searchParams: Record<string, string | string[] | undefined>) {
  const code = typeof searchParams.error === "string" ? searchParams.error : "";
  const message = typeof searchParams.message === "string" ? searchParams.message : "";

  switch (code) {
    case "oauth":
      return `Google sign-in could not be completed.${message ? ` ${message}` : ""}`;
    case "userinfo":
      return "Google sign-in completed, but user details could not be retrieved.";
    case "db":
      return `Account setup failed.${message ? ` ${message}` : ""}`;
    default:
      return "";
  }
}

export default async function HomePage({ searchParams }: HomePageProps) {
  const resolvedSearchParams = (await searchParams) || {};
  const error = readError(resolvedSearchParams);

  return (
    <main className="login-page">
      <section className="login-hero">
        <div className="login-hero-photo" />
        <div className="login-hero-grid" />

        <div className="login-hero-copy">
          <span className="eyebrow-pill">BNS Legal Intelligence Desk</span>
          <h1>Indian criminal law research for police, courts, and investigators.</h1>
          <p>
            A modern legal intelligence workspace for Bharatiya Nyaya Sanhita, Bharatiya
            Nagarik Suraksha Sanhita, and Bharatiya Sakshya Adhiniyam. Designed like a
            digital police-legal command desk instead of a generic AI chat tool.
          </p>

          <div className="hero-stat-grid">
            <article className="glass-card">
              <strong>Legal Research</strong>
              <span>Search provisions, compare sections, and explain offences quickly.</span>
            </article>
            <article className="glass-card">
              <strong>Investigation Ready</strong>
              <span>Built for case review, arrest procedure, evidence flow, and legal brief preparation.</span>
            </article>
            <article className="glass-card">
              <strong>Institutional Design</strong>
              <span>Premium light styling inspired by Indian courts, police desks, and statute files.</span>
            </article>
          </div>
        </div>

        <aside className="hero-side-brief">
          <h3>Operational Snapshot</h3>
          <div className="hero-brief-list">
            <div className="hero-brief-item">
              <strong>BNS</strong>
              <span>Substantive criminal law analysis for offences, ingredients, and punishments.</span>
            </div>
            <div className="hero-brief-item">
              <strong>BNSS + BSA</strong>
              <span>Procedure and evidence support for arrest, trial stages, statements, and proof.</span>
            </div>
          </div>
        </aside>
      </section>

      <section className="login-panel-wrap">
        <div className="login-panel">
          <div className="login-panel-head">
            <div className="seal-mark">AI</div>
            <div>
              <h2>Secure Access</h2>
              <p>Enter the legal intelligence workspace used for BNS, BNSS, and BSA query support.</p>
            </div>
          </div>

          <div className="trust-pill-row">
            <span className="trust-pill">Police and legal workflow</span>
            <span className="trust-pill">Google-authenticated entry</span>
            <span className="trust-pill">Saved research sessions</span>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}

          <div className="feature-stack">
            <article className="feature-card">
              <div className="feature-icon">⚖</div>
              <div>
                <h3>BNS legal brief generation</h3>
                <p>Structured plain-language explanations, statutory elements, exceptions, and consequences.</p>
              </div>
            </article>

            <article className="feature-card">
              <div className="feature-icon">⌕</div>
              <div>
                <h3>Fast statutory lookup</h3>
                <p>Search BNS, BNSS, and BSA provisions with a workflow suited to police and legal review.</p>
              </div>
            </article>

            <article className="feature-card">
              <div className="feature-icon">▣</div>
              <div>
                <h3>Persistent investigation history</h3>
                <p>Return to past legal scenarios, case questions, and ongoing research sessions from one workspace.</p>
              </div>
            </article>
          </div>

          <div className="divider-label">Authorized Sign In</div>

          <a className="primary-auth-button" href={`${API_BASE_URL}/auth/google`}>
            Continue with Google
          </a>

          <p className="panel-foot-copy">
            <strong>Private research workspace.</strong> Sessions are tied to your account so officers,
            advocates, researchers, and students can continue work without losing context.
          </p>
        </div>
      </section>
    </main>
  );
}
