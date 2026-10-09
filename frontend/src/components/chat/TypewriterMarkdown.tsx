import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
  speedMs?: number;
  onComplete?: () => void;
}

export function TypewriterMarkdown({ content, speedMs = 15, onComplete }: Props) {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    if (!content) {
      setDisplayedText("");
      setIsTyping(false);
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

  return (
    <div className="relative font-sans text-xs text-neutral-200 leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayedText}</ReactMarkdown>
      {isTyping && (
        <span className="inline-block w-1.5 h-3.5 ml-1 bg-emerald-400 animate-pulse rounded-sm vertical-middle" />
      )}
    </div>
  );
}
