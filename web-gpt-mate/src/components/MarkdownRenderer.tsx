import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { cn } from "@/lib/utils";

interface MarkdownRendererProps {
  content: string;
  className?: string;
  toolState?: { id_to_iframe?: Record<string, string> };
}

const MarkdownRenderer = ({ content, className, toolState }: MarkdownRendererProps) => {
  // 【숫자†chart】 형식의 차트 ID를 iframe으로 변환
  const processContent = (text: string): string => {
    if (!toolState?.id_to_iframe) {
      return text;
    }
    
    // 【숫자†chart】 패턴 찾기
    const chartPattern = /【(\d+†chart)】/g;
    return text.replace(chartPattern, (match, key) => {
      const iframeHtml = toolState.id_to_iframe[key];
      if (iframeHtml) {
        // iframe HTML을 그대로 삽입 (rehypeRaw가 파싱할 수 있도록)
        // 줄바꿈을 추가하여 마크다운 파서가 제대로 인식하도록 함
        return `\n\n${iframeHtml}\n\n`;
      }
      return match;
    });
  };

  const processedContent = processContent(content);

  return (
    <div className={cn("prose prose-sm max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
        code({ node, inline, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <SyntaxHighlighter
              style={vscDarkPlus}
              language={match[1]}
              PreTag="div"
              className="rounded-md my-2"
              {...props}
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code
              className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono"
              {...props}
            >
              {children}
            </code>
          );
        },
        h1: ({ children }) => (
          <h1 className="text-2xl font-bold mt-6 mb-4">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-bold mt-5 mb-3">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-bold mt-4 mb-2">{children}</h3>
        ),
        p: ({ children }) => <p className="mb-4 leading-7">{children}</p>,
        ul: ({ children }) => (
          <ul className="list-disc list-inside mb-4 space-y-2">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-inside mb-4 space-y-2">
            {children}
          </ol>
        ),
        li: ({ children }) => <li className="leading-7">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-primary pl-4 italic my-4">
            {children}
          </blockquote>
        ),
        a: ({ children, href }) => (
          <a
            href={href}
            className="text-primary hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            {children}
          </a>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border-collapse border border-border">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-border px-4 py-2 bg-muted font-semibold text-left">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-border px-4 py-2">{children}</td>
        ),
        // iframe 렌더링을 위한 커스텀 컴포넌트
        iframe: ({ src, ...props }: any) => (
          <div className="my-4 rounded-lg overflow-hidden border border-border">
            <iframe
              src={src}
              className="w-full aspect-video"
              allowFullScreen
              {...props}
            />
          </div>
        ),
      }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
