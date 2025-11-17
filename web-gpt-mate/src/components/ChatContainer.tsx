import { useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "@/types/chat";

interface ChatContainerProps {
  messages: Message[];
  isLoading?: boolean;
  toolState?: { id_to_iframe?: Record<string, string> };
}

const ChatContainer = ({ messages, isLoading, toolState }: ChatContainerProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <ScrollArea className="flex-1">
      <div ref={scrollRef} className="max-w-4xl mx-auto">
        {isLoading && (
          <div className="flex justify-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
          </div>
        )}
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full py-20 px-4">
            <div className="text-center space-y-4">
              <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                AI 채팅 어시스턴트
              </h1>
              <p className="text-muted-foreground text-lg">
                무엇을 도와드릴까요?
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-8 max-w-2xl">
                <div className="p-4 rounded-lg bg-card border hover:border-primary transition-colors cursor-pointer">
                  <p className="text-sm font-medium">💡 아이디어 브레인스토밍</p>
                </div>
                <div className="p-4 rounded-lg bg-card border hover:border-primary transition-colors cursor-pointer">
                  <p className="text-sm font-medium">📝 글쓰기 도움</p>
                </div>
                <div className="p-4 rounded-lg bg-card border hover:border-primary transition-colors cursor-pointer">
                  <p className="text-sm font-medium">🔍 정보 검색</p>
                </div>
                <div className="p-4 rounded-lg bg-card border hover:border-primary transition-colors cursor-pointer">
                  <p className="text-sm font-medium">💻 코딩 지원</p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-0">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message.text}
                isUser={message.isUser}
                timestamp={message.timestamp}
                reasoningSteps={message.reasoningSteps}
                isThinking={message.isThinking}
                toolState={toolState}
              />
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  );
};

export default ChatContainer;
