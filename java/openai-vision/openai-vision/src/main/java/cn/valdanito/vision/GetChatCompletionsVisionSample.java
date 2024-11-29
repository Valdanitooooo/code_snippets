package cn.valdanito.vision;

import io.github.sashirestela.openai.SimpleOpenAI;
import io.github.sashirestela.openai.common.content.ContentPart;
import io.github.sashirestela.openai.common.content.ContentPart.ContentPartImageUrl.ImageUrl;
import io.github.sashirestela.openai.domain.chat.ChatMessage;
import io.github.sashirestela.openai.domain.chat.ChatRequest;
import io.github.sashirestela.openai.domain.chat.Chat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.List;

public class GetChatCompletionsVisionSample {

    private static ImageUrl loadImageAsBase64(String imagePath) {
        try {
            Path path = Paths.get(imagePath);
            byte[] imageBytes = Files.readAllBytes(path);
            String base64String = Base64.getEncoder().encodeToString(imageBytes);
            var extension = imagePath.substring(imagePath.lastIndexOf('.') + 1);
            var prefix = "data:image/" + extension + ";base64,";
            return ImageUrl.of(prefix + base64String);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        //在线图片
//        var imageUrl = ImageUrl.of(
//                "http://xxxxxx")
//                                );
        //本地图片
        var imageUrl = loadImageAsBase64("/Users/valdanito/Downloads/test111.jpg");
        var openai = SimpleOpenAI.builder()
                .baseUrl("http://xxxx:5000")
                .apiKey("xxx")
                .build();
        var chatRequest = ChatRequest.builder()
                .model("Qwen2-VL-7B-Instruct")
                .messages(
                        List.of(
                                ChatMessage.UserMessage.of(
                                        List.of(
                                                ContentPart.ContentPartText.of(
                                                        "图片内容是什么"),
                                                ContentPart.ContentPartImageUrl.of(
                                                        imageUrl
                                                )
                                        )
                                )
                        )
                )
                .temperature(0.0)
                .maxTokens(8192)
                .build();
        var chatResponse = openai.chatCompletions().createStream(chatRequest).join();
        chatResponse.filter(
                        resp -> resp.getChoices().size() > 0 && resp.firstContent() != null
                )
                .map(Chat::firstContent)
                .forEach(System.out::print);
        System.out.println();
    }
}
