import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { Bot, ChevronDown, Loader2, Sparkles, User } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { ReasoningStep } from "@/types/chat";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp?: Date;
  reasoningSteps?: ReasoningStep[];
  isThinking?: boolean;
  toolState?: { id_to_iframe?: Record<string, string> };
}

const stageLabels: Record<string, string> = {
  router: "요청 분석",
  query_refinement: "검색어 최적화",
  search: "웹 검색",
  summary: "결과 요약",
  final: "최종 응답",
};

const ChatMessage = ({ message, isUser, timestamp, reasoningSteps = [], isThinking = false, toolState }: ChatMessageProps) => {
  const [open, setOpen] = useState(false);
  const hasReasoning = reasoningSteps.length > 0;
  const showToggle = !isUser && (isThinking || hasReasoning);

  const sortedSteps = useMemo(
    () => [...reasoningSteps].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()),
    [reasoningSteps]
  );

  return (
    <div
      className={cn(
        "flex gap-4 p-6 animate-in fade-in slide-in-from-bottom-3 duration-500",
        isUser ? "bg-background" : "bg-muted/30"
      )}
    >
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <User className="w-5 h-5 text-primary-foreground" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center">
            <Bot className="w-5 h-5 text-accent-foreground" />
          </div>
        )}
      </div>
      <div className="flex-1 space-y-2 overflow-hidden">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">
            {isUser ? "You" : "AI Assistant"}
          </span>
          {timestamp && (
            <span className="text-xs text-muted-foreground">
              {timestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          )}
        </div>
        <div className="text-sm leading-relaxed overflow-hidden">
          {isUser ? (
            <div className="whitespace-pre-wrap break-words">{message}</div>
          ) : (
            <MarkdownRenderer content={message} toolState={toolState} />
          )}
        </div>

        {showToggle && (
          <Collapsible open={open} onOpenChange={setOpen} className="pt-2">
            <div className="flex items-center gap-2">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 px-2 text-xs font-medium">
                  {isThinking ? (
                    <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="mr-2 h-3.5 w-3.5" />
                  )}
                  생각 과정 보기
                  <ChevronDown
                    className={cn(
                      "ml-1 h-3 w-3 transition-transform",
                      open ? "rotate-180" : "rotate-0"
                    )}
                  />
                </Button>
              </CollapsibleTrigger>
              {hasReasoning && (
                <span className="text-xs text-muted-foreground">{sortedSteps.length} 단계</span>
              )}
            </div>
            <CollapsibleContent className="mt-2 space-y-2">
              {isThinking && sortedSteps.length === 0 ? (
                <div className="flex items-center gap-2 rounded-md border bg-background p-3 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>검색 전략을 고민하고 있어요...</span>
                </div>
              ) : (
                sortedSteps.map((step) => {
                  const label = stageLabels[step.stage] ?? step.stage;
                  return (
                    <div key={step.id} className="rounded-md border bg-background p-3 text-sm shadow-sm">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span className="font-semibold">{label}</span>
                        <span>
                          {step.iteration ? `#${step.iteration}` : ""}
                          {step.iteration ? " · " : ""}
                          {step.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                      <p className="mt-2 whitespace-pre-wrap leading-relaxed text-sm">{step.message}</p>
                    </div>
                  );
                })
              )}
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
