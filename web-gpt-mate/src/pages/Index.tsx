import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ChatContainer from "@/components/ChatContainer";
import ChatInput from "@/components/ChatInput";
import { AppSidebar, Conversation } from "@/components/AppSidebar";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { useToast } from "@/hooks/use-toast";
import { streamLangGraphChat, getChatHistory, deleteChatHistory } from "@/lib/api";
import { Message, ReasoningStep } from "@/types/chat";

const Index = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<
    string | undefined
  >();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [toolState, setToolState] = useState<{ id_to_iframe?: Record<string, string> }>({});
  const { toast } = useToast();
  const navigate = useNavigate();
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancelOngoingStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  // 컴포넌트 마운트 시 자동으로 새 대화 생성
  useEffect(() => {
    if (conversations.length === 0 && !currentConversationId) {
      handleNewConversation();
    }
  }, []);

  // 대화 선택 시 히스토리 로드
  useEffect(() => {
    if (currentConversationId) {
      loadChatHistory(currentConversationId);
    }
  }, [currentConversationId]);

  const loadChatHistory = async (chatId: string) => {
    try {
      setIsLoadingHistory(true);
      const response = await getChatHistory(chatId);
      
      if (response.messages && Array.isArray(response.messages)) {
        // 백엔드 메시지 형식을 프론트엔드 형식으로 변환
        const loadedMessages: Message[] = response.messages.map((msg: any, idx: number) => ({
          id: `${chatId}-${idx}`,
          text: msg.content || "",
          isUser: msg.role === "user",
          timestamp: new Date(),
          reasoningSteps: [],
          isThinking: false,
        }));
        setMessages(loadedMessages);
      } else {
        setMessages([]);
      }
    } catch (error) {
      console.error("Failed to load chat history:", error);
      setMessages([]);
      toast({
        title: "히스토리 로드 실패",
        description: error instanceof Error ? error.message : "대화 내용을 불러올 수 없습니다.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleNewConversation = () => {
    cancelOngoingStream();
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "새 대화",
      timestamp: new Date(),
    };
    setConversations((prev) => [newConversation, ...prev]);
    setCurrentConversationId(newConversation.id);
    setMessages([]);
  };

  const handleSelectConversation = (id: string) => {
    cancelOngoingStream();
    setCurrentConversationId(id);
    // 히스토리는 useEffect에서 자동으로 로드됨
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      cancelOngoingStream();
      await deleteChatHistory(id);
      setConversations((prev) => prev.filter((conv) => conv.id !== id));
      if (currentConversationId === id) {
        setCurrentConversationId(undefined);
        setMessages([]);
      }
      toast({
        title: "대화 삭제됨",
        description: "대화가 삭제되었습니다.",
      });
    } catch (error) {
      toast({
        title: "삭제 실패",
        description: error instanceof Error ? error.message : "대화를 삭제할 수 없습니다.",
        variant: "destructive",
      });
    }
  };

  const handleSendMessage = async (text: string) => {
    // 이전 요청이 있으면 취소
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // 대화가 없으면 새로 생성
    let chatId = currentConversationId;
    if (!chatId) {
      chatId = Date.now().toString();
      const newConversation: Conversation = {
        id: chatId,
        title: text.slice(0, 30) + (text.length > 30 ? "..." : ""),
        lastMessage: text.slice(0, 50),
        timestamp: new Date(),
      };
      setConversations((prev) => [newConversation, ...prev]);
      setCurrentConversationId(chatId);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // 대화 목록 업데이트
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === chatId
          ? { ...conv, lastMessage: text.slice(0, 50), timestamp: new Date() }
          : conv
      )
    );

    // AI 응답 메시지 초기화
    const aiMessageId = (Date.now() + 1).toString();
    const aiMessage: Message = {
      id: aiMessageId,
      text: "",
      isUser: false,
      timestamp: new Date(),
      reasoningSteps: [],
      isThinking: true,
    };
    setMessages((prev) => [...prev, aiMessage]);

    try {
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const stream = streamLangGraphChat(
        {
          question: text,
          chatId: chatId,
          userInfo: null,
        },
        controller.signal
      );

      let accumulatedText = "";

      for await (const event of stream) {
        if (event.event === "token") {
          const chunk = typeof event.data === "string" ? event.data : "";
          for (const char of chunk) {
            accumulatedText += char;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiMessageId
                  ? { ...msg, text: accumulatedText }
                  : msg
              )
            );
          }
        } else if (event.event === "reasoning") {
          const payload = event.data ?? {};
          const stage = typeof payload?.stage === "string" ? payload.stage : "info";
          const message =
            typeof payload === "string"
              ? payload
              : typeof payload?.message === "string"
              ? payload.message
              : JSON.stringify(payload);
          const iteration =
            typeof payload === "object" && payload !== null && typeof payload.iteration === "number"
              ? payload.iteration
              : undefined;
          const step: ReasoningStep = {
            id: `${aiMessageId}-reason-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            stage,
            message,
            iteration,
            timestamp: new Date(),
          };
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? {
                    ...msg,
                    reasoningSteps: [...(msg.reasoningSteps ?? []), step],
                  }
                : msg
            )
          );
        } else if (event.event === "tool_state") {
          // tool_state 업데이트 (차트, DA 시각화 등)
          if (event.data && typeof event.data === "object") {
            setToolState(event.data);
          }
        } else if (event.event === "metadata") {
          console.debug("LangGraph metadata", event.data);
        } else if (event.event === "error") {
          // 에러 처리
          const errorText = event.data || "알 수 없는 오류가 발생했습니다.";
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, text: `오류: ${errorText}` }
                : msg
            )
          );
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, isThinking: false }
                : msg
            )
          );
          toast({
            title: "오류 발생",
            description: errorText,
            variant: "destructive",
          });
          break;
        } else if (event.event === "result") {
          setConversations((prev) =>
            prev.map((conv) =>
              conv.id === chatId
                ? { ...conv, lastMessage: accumulatedText.slice(0, 80), timestamp: new Date() }
                : conv
            )
          );
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, isThinking: false }
                : msg
            )
          );
          break;
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMessageId
              ? { ...msg, isThinking: false }
              : msg
          )
        );
        return;
      }
      console.error("Chat stream error:", error);
      const errorMessage =
        error instanceof Error ? error.message : "연결 오류가 발생했습니다.";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? { ...msg, text: `오류: ${errorMessage}` }
            : msg
        )
      );
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? { ...msg, isThinking: false }
            : msg
        )
      );
      toast({
        title: "연결 오류",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? { ...msg, isThinking: false }
            : msg
        )
      );
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated");
    localStorage.removeItem("username");
    navigate("/login");
  };

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-chat-background">
        <AppSidebar
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onLogout={handleLogout}
        />

        <div className="flex flex-col flex-1">
          <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="flex items-center px-4 py-4 gap-3">
              <SidebarTrigger />
              <h1 className="text-xl font-semibold">AI Chat - Multi-turn</h1>
              {isLoadingHistory && <span className="text-xs text-muted-foreground">로딩 중...</span>}
            </div>
          </header>

          <ChatContainer messages={messages} isLoading={isLoadingHistory} toolState={toolState} />

          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading || isLoadingHistory} />
        </div>
      </div>
    </SidebarProvider>
  );
};

export default Index;
