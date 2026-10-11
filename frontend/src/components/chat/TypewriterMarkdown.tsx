import { useEffect, useState, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
  speedMs?: number;
  validCitations?: string[];
  onCitationClick?: (citation: string) => void;
  onComplete?: () => void;
}

export function TypewriterMarkdown({
  content,
  speedMs = 15,
  validCitations,
  onCitationClick,
  onComplete,
}: Props) {
  const [displayedText, setDisplayedText] = useState(() => (speedMs <= 0 ? content : ""));
  const [isTyping, setIsTyping] = useState(() => (speedMs <= 0 ? false : !!content));

  useEffect(() => {
    if (!content) {
      setDisplayedText("");
      setIsTyping(false);
      return;
    }

    if (speedMs <= 0) {
      setDisplayedText(content);
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    // Incremental typewriter animation
    let currentIndex = 0;
    const chunkSize = 4; // Reveal 4 characters per tick for smooth natural streaming
    setIsTyping(true);

    const interval = setInterval(() => {
      currentIndex += chunkSize;
      if (currentIndex >= content.length) {
        setDisplayedText(content);
        setIsTyping(false);
        clearInterval(interval);
        if (onComplete) onComplete();
      } else {
        setDisplayedText(content.slice(0, currentIndex));
      }
    }, speedMs);

    return () => clearInterval(interval);
  }, [content, speedMs, onComplete]);

  // Transform bracketed citations into markdown links with citation: scheme
  const processedMarkdown = useMemo(() => {
    if (!displayedText) return "";
    return displayedText.replace(
      /\[(web:[a-zA-Z0-9_\-]+|(?:[^\]\n]+,\s*art\s*[^\]\n]+))\]/g,
      (_match, p1) => `[${p1}](citation:${encodeURIComponent(p1)})`
    );
  }, [displayedText]);

  return (
    <div className="relative font-sans text-xs text-neutral-200 leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => url}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("citation:")) {
              const citationTarget = decodeURIComponent(href.replace(/^citation:/, ""));
              const isVerified =
                validCitations && validCitations.length > 0
                  ? validCitations.some((c) => {
                      const cNorm = c.toLowerCase().trim().replace(/^\[|\]$/g, "");
                      const targetNorm = citationTarget.toLowerCase().trim().replace(/^\[|\]$/g, "");
                      return cNorm.includes(targetNorm) || targetNorm.includes(cNorm);
                    })
                  : true; // If no explicit verification list passed, allow opening passage modal

              return (
                <button
                  type="button"
                  onClick={() => onCitationClick && onCitationClick(citationTarget)}
                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded text-[10px] font-mono cursor-pointer transition-colors align-baseline ${
                    !isVerified
                      ? "bg-amber-950/70 text-amber-300 border border-amber-500/40 hover:bg-amber-900/80"
                      : "bg-emerald-950/60 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-900/70"
                  }`}
                  title={
                    !isVerified
                      ? "Unverified citation - click to inspect"
                      : "Click to view stored evidence passage"
                  }
                >
                  [{children}]
                  {!isVerified && (
                    <span className="text-[8px] px-1 rounded bg-amber-500/20 text-amber-300 font-sans uppercase font-bold tracking-tight">
                      unverified
                    </span>
                  )}
                </button>
              );
            }

            // Normal external links: enforce http/https only and inert plain text for dangerous schemes
            const isSafeScheme = href && /^https?:\/\//i.test(href);
            if (!isSafeScheme) {
              return <span>{children}</span>;
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer nofollow"
                className="text-cyan-400 hover:text-cyan-300 underline"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {processedMarkdown}
      </ReactMarkdown>
      {isTyping && (
        <span className="inline-block w-1.5 h-3.5 ml-1 bg-emerald-400 animate-pulse rounded-sm align-middle" />
      )}
    </div>
  );
}
