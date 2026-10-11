import { useState } from "react";
import { Globe, Search, ExternalLink, ChevronDown, ChevronRight, Compass } from "lucide-react";
import type { ExecutionEvent } from "@/types/execution";

export interface WebSource {
  url: string;
  title: string;
  domain: string;
  snippet: string;
  faviconUrl: string;
}

interface Props {
  events: ExecutionEvent[];
  running: boolean;
}

function extractDomain(urlStr: string): string {
  try {
    const parsed = new URL(urlStr);
    return parsed.hostname.replace(/^www\./, "");
  } catch {
    return urlStr;
  }
}

function getFaviconUrl(domain: string): string {
  return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=64`;
}

export function WebSearchIndicator({ events, running }: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [failedFavicons, setFailedFavicons] = useState<Record<string, boolean>>({});

  // Filter and extract web events
  const webEvents = events.filter((ev) => {
    const t = ev.rawType || ev.label;
    return (
      t === "WEB_SEARCH_STARTED" ||
      t === "WEB_SEARCH_COMPLETED" ||
      t === "WEB_FETCH_STARTED" ||
      t === "WEB_FETCH_COMPLETED"
    );
  });

  if (webEvents.length === 0 && !running) return null;

  // Extract all queries searched
  const queries: string[] = [];
  let latestQuery = "";
  let currentlyViewing: { url: string; title: string; domain: string } | null = null;
  const sourcesMap = new Map<string, WebSource>();

  for (const ev of webEvents) {
    const t = ev.rawType || ev.label;
    const payload = ev.payload || {};

    if (t === "WEB_SEARCH_STARTED") {
      const q = (payload.query as string) || ev.message?.replace(/^Searching the web for:\s*"?/, "").replace(/"?$/, "") || "";
      if (q && !queries.includes(q)) queries.push(q);
      latestQuery = q;
    } else if (t === "WEB_SEARCH_COMPLETED") {
      const results = (payload.results as Array<{ title?: string; url?: string; snippet?: string }>) || [];
      for (const res of results) {
        if (res.url) {
          const dom = extractDomain(res.url);
          sourcesMap.set(res.url, {
            url: res.url,
            title: res.title || dom,
            domain: dom,
            snippet: res.snippet || "",
            faviconUrl: getFaviconUrl(dom),
          });
        }
      }
    } else if (t === "WEB_FETCH_STARTED") {
      const url = (payload.url as string) || "";
      const title = (payload.title as string) || "";
      if (url) {
        const dom = extractDomain(url);
        currentlyViewing = { url, title: title || dom, domain: dom };
      }
    } else if (t === "WEB_FETCH_COMPLETED") {
      currentlyViewing = null;
      const url = (payload.url as string) || "";
      const title = (payload.title as string) || "";
      const snippet = (payload.snippet as string) || "";
      if (url) {
        const dom = extractDomain(url);
        const existing = sourcesMap.get(url);
        sourcesMap.set(url, {
          url,
          title: title || existing?.title || dom,
          domain: dom,
          snippet: snippet || existing?.snippet || "",
          faviconUrl: getFaviconUrl(dom),
        });
      }
    }
  }

  const sources = Array.from(sourcesMap.values());
  const hasActiveWebActivity = running && webEvents.length > 0;

  // If there are no web events and we are running, check if anything web-related has fired
  if (webEvents.length === 0) return null;

  const handleFaviconError = (domain: string) => {
    setFailedFavicons((prev) => ({ ...prev, [domain]: true }));
  };

  return (
    <div className="rounded-xl border border-cyan-500/20 bg-neutral-950/90 overflow-hidden shadow-lg shadow-cyan-950/10 transition-all my-2.5 backdrop-blur-sm">
      {/* Active Live Indicator Bar */}
      {hasActiveWebActivity ? (
        <div className="p-3 bg-gradient-to-r from-cyan-950/40 via-blue-950/30 to-neutral-900/60 border-b border-cyan-500/20">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="relative flex items-center justify-center w-6 h-6 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex-shrink-0">
                <Globe size={13} className="text-cyan-400 animate-spin" style={{ animationDuration: "6s" }} />
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-medium text-cyan-200 truncate">
                    {currentlyViewing ? (
                      <>
                        Viewing <span className="font-semibold text-white underline decoration-cyan-400/50 underline-offset-2">{currentlyViewing.title || currentlyViewing.domain}</span>
                      </>
                    ) : latestQuery ? (
                      <>
                        Searching the web for <span className="font-semibold text-white font-mono">"{latestQuery}"</span>
                      </>
                    ) : (
                      "Searching the web for live sources..."
                    )}
                  </span>
                </div>
                <p className="text-[10px] text-cyan-400/70 font-mono flex items-center gap-1.5 mt-0.5">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  Live Web Search • Tavily Intelligence
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 animate-pulse flex items-center gap-1">
                <Search size={10} className="animate-spin text-cyan-400" />
                SEARCHING
              </span>
            </div>
          </div>
        </div>
      ) : (
        /* Finished State Header (ChatGPT / Claude Style Collapsible Bar) */
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between px-3.5 py-2.5 bg-neutral-900/60 hover:bg-neutral-900/90 text-left transition-colors cursor-pointer group"
        >
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-5 h-5 rounded-md bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center flex-shrink-0 text-cyan-400">
              <Compass size={13} />
            </div>
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-medium text-xs text-neutral-200 group-hover:text-cyan-300 transition-colors">
                Searched {sources.length} {sources.length === 1 ? "website" : "websites"}
              </span>
              {/* Domain Favicon Pills Preview in Header */}
              <div className="hidden sm:flex items-center gap-1 pl-1.5 border-l border-white/10">
                {sources.slice(0, 4).map((s, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-neutral-300 font-mono"
                    title={s.title}
                  >
                    {!failedFavicons[s.domain] ? (
                      <img
                        src={s.faviconUrl}
                        alt=""
                        className="w-3 h-3 rounded-sm object-contain"
                        onError={() => handleFaviconError(s.domain)}
                      />
                    ) : (
                      <Globe size={10} className="text-neutral-400" />
                    )}
                    <span className="truncate max-w-[90px]">{s.domain}</span>
                  </span>
                ))}
                {sources.length > 4 && (
                  <span className="text-[10px] text-neutral-400 font-mono px-1">
                    +{sources.length - 4} more
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 text-neutral-400">
            <span className="text-[10px] font-mono text-cyan-400/80 uppercase tracking-wider">
              {isOpen ? "Hide sources" : "View sources"}
            </span>
            {isOpen ? <ChevronDown size={14} className="text-neutral-400 group-hover:text-white" /> : <ChevronRight size={14} className="text-neutral-400 group-hover:text-white" />}
          </div>
        </button>
      )}

      {/* Website Preview Cards Grid */}
      {isOpen && sources.length > 0 && (
        <div className="p-3 border-t border-cyan-500/10 bg-neutral-950/60">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
            {sources.map((source, index) => (
              <a
                key={index}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group relative flex flex-col justify-between p-3 rounded-xl bg-neutral-900/60 hover:bg-neutral-900 border border-white/10 hover:border-cyan-500/40 transition-all duration-150 shadow-sm hover:shadow-cyan-950/30 overflow-hidden"
              >
                <div>
                  {/* Card Header: Favicon + Domain + External Link */}
                  <div className="flex items-center justify-between gap-1.5 mb-1.5">
                    <div className="flex items-center gap-1.5 min-w-0">
                      {!failedFavicons[source.domain] ? (
                        <img
                          src={source.faviconUrl}
                          alt=""
                          className="w-3.5 h-3.5 rounded object-contain flex-shrink-0"
                          onError={() => handleFaviconError(source.domain)}
                        />
                      ) : (
                        <Globe size={13} className="text-cyan-400 flex-shrink-0" />
                      )}
                      <span className="text-[10px] font-mono text-neutral-400 truncate group-hover:text-cyan-300 transition-colors">
                        {source.domain}
                      </span>
                    </div>
                    <ExternalLink size={11} className="text-neutral-500 group-hover:text-cyan-400 flex-shrink-0 transition-colors" />
                  </div>

                  {/* Website Title */}
                  <h4 className="text-xs font-semibold text-neutral-200 group-hover:text-white line-clamp-2 leading-snug transition-colors">
                    {source.title}
                  </h4>

                  {/* Excerpt / Snippet Preview */}
                  {source.snippet && (
                    <p className="text-[11px] text-neutral-400 line-clamp-2 mt-1.5 leading-relaxed font-sans">
                      {source.snippet}
                    </p>
                  )}
                </div>

                {/* Sub-label badge */}
                <div className="mt-2.5 pt-2 border-t border-white/5 flex items-center justify-between text-[9px] font-mono text-neutral-400">
                  <span className="text-neutral-400 group-hover:text-cyan-400 transition-colors">
                    Source #{index + 1}
                  </span>
                  <span className="opacity-0 group-hover:opacity-100 text-cyan-400 flex items-center gap-0.5 transition-opacity">
                    Open page ›
                  </span>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
