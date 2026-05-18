package com.griddb.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;

import java.util.*;

/**
 * POST /api/ai/suggest
 *
 * Accepts a compact RCA summary payload from the browser,
 * calls the Gemini 1.5 Flash API, and returns 4-5 bullet-point suggestions.
 *
 * Expected request body (all fields optional except "defectTable"):
 * {
 *   "defectTable":    "my_table",
 *   "totalDefects":   1234,
 *   "spikeWindows":   3,
 *   "topDefectTypes": "ScratchA×45, BubbleB×30, CrackC×12",
 *   "topCorrelations":"temp_C r=+0.82, pressure_bar r=+0.61, operator_id r=-0.55",
 *   "capaKPIs":       "recurrence=18%, effectiveness=42%, escapeRate=3%, containment=6h",
 *   "context":        "automotive body panel stamping line, 3-shift operation"
 * }
 */
@RestController
@RequestMapping("/api/ai")
public class GeminiController {

    @Value("${gemini.api.key:}")
    private String apiKey;

    private static final String GEMINI_URL =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key=";

    private final RestTemplate restTemplate = new RestTemplate();

    @PostMapping("/suggest")
    public ResponseEntity<Map<String, Object>> suggest(@RequestBody Map<String, Object> payload) {
        if (apiKey == null || apiKey.isBlank()) {
            return ResponseEntity.status(503)
                .body(Map.of("error", "Gemini API key not configured on server"));
        }

        String prompt = buildPrompt(payload);

        Map<String, Object> genConfig = new LinkedHashMap<>();
        genConfig.put("maxOutputTokens", 200);
        genConfig.put("temperature", 0.25);

        Map<String, Object> requestBody = Map.of(
            "contents", List.of(Map.of(
                "parts", List.of(Map.of("text", prompt))
            )),
            "generationConfig", genConfig
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        try {
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = (ResponseEntity<Map<String, Object>>)
                (ResponseEntity<?>) restTemplate.exchange(
                    GEMINI_URL + apiKey,
                    HttpMethod.POST,
                    new HttpEntity<>(requestBody, headers),
                    Map.class
                );

            String text = extractText(response.getBody());
            return ResponseEntity.ok(Map.of("suggestions", text != null ? text : "No suggestions generated."));

        } catch (HttpClientErrorException.TooManyRequests e) {
            // Surface retry delay to the client rather than swallowing it as 500
            return ResponseEntity.status(429)
                .body(Map.of("error", "Gemini rate limit reached — free-tier quota exhausted. " +
                    "Please wait ~60 s and try again, or enable billing at https://ai.dev/rate-limit"));
        } catch (Exception e) {
            return ResponseEntity.status(500)
                .body(Map.of("error", "Gemini call failed: " + e.getMessage()));
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private String extractText(Map<String, Object> body) {
        if (body == null) return null;
        Object candidates = body.get("candidates");
        if (!(candidates instanceof List<?> cList) || cList.isEmpty()) return null;
        Object first = cList.get(0);
        if (!(first instanceof Map<?, ?> cand)) return null;
        Object content = cand.get("content");
        if (!(content instanceof Map<?, ?> contentMap)) return null;
        Object parts = contentMap.get("parts");
        if (!(parts instanceof List<?> pList) || pList.isEmpty()) return null;
        Object part = pList.get(0);
        if (!(part instanceof Map<?, ?> partMap)) return null;
        return (String) partMap.get("text");
    }

    private String buildPrompt(Map<String, Object> p) {
        StringBuilder sb = new StringBuilder();

        // Hard rules first — Gemini must follow these before reading the data
        sb.append("STRICT RULES (violating any rule makes the response worthless):\n");
        sb.append("1. Output EXACTLY 4 bullet points. No more, no less.\n");
        sb.append("2. Every bullet MUST quote a specific number, column name, or metric from DATA below.\n");
        sb.append("3. Zero generic advice. Zero filler. No 'consider reviewing'. No 'it is recommended'.\n");
        sb.append("4. Format every bullet as: \u2022 [exact data observation] \u2192 [one specific action, \u226412 words]\n");
        sb.append("5. No intro sentence. No conclusion. Start directly with the first \u2022\n\n");

        sb.append("DATA (ground truth \u2014 do not invent any value not listed here):\n");

        if (p.get("defectTable")     != null) sb.append("table=").append(p.get("defectTable")).append("\n");
        if (p.get("totalRows")       != null) sb.append("total_rows=").append(p.get("totalRows")).append("\n");
        if (p.get("totalDefects")    != null) sb.append("total_defects=").append(p.get("totalDefects")).append("\n");
        if (p.get("spikeWindows")    != null) sb.append("spike_windows=").append(p.get("spikeWindows")).append("\n");
        if (p.get("worstSpike")      != null) sb.append("worst_spike=").append(p.get("worstSpike")).append("\n");
        if (p.get("topDefectTypes")  != null) sb.append("top_defect_types=").append(p.get("topDefectTypes")).append("\n");
        if (p.get("topCorrelations") != null) sb.append("correlations=").append(p.get("topCorrelations")).append("\n");
        if (p.get("lowConfCorr")     != null) sb.append("low_confidence_correlations=").append(p.get("lowConfCorr")).append("\n");
        if (p.get("capaKPIs")        != null) sb.append("capa_kpis=").append(p.get("capaKPIs")).append("\n");
        if (p.get("capaFailures")    != null) sb.append("capa_threshold_failures=").append(p.get("capaFailures")).append("\n");

        sb.append("\nOutput 4 bullets now:");
        return sb.toString();
    }
}
