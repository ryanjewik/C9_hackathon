package com.example.data_service.dto;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Converter(autoApply = false)
public class PostgresIntegerArrayConverter implements AttributeConverter<List<Integer>, String> {

    @Override
    public String convertToDatabaseColumn(List<Integer> attribute) {
        if (attribute == null) return null;
        if (attribute.isEmpty()) return "{}";
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        for (int i = 0; i < attribute.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(attribute.get(i));
        }
        sb.append('}');
        return sb.toString();
    }

    @Override
    public List<Integer> convertToEntityAttribute(String dbData) {
        if (dbData == null) return null;
        dbData = dbData.trim();
        if (dbData.equals("{}")) return Collections.emptyList();
        if (dbData.startsWith("{") && dbData.endsWith("}")) {
            dbData = dbData.substring(1, dbData.length() - 1);
        }
        if (dbData.isBlank()) return Collections.emptyList();
        String[] parts = dbData.split(",");
        List<Integer> out = new ArrayList<>(parts.length);
        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            try {
                out.add(Integer.valueOf(p));
            } catch (NumberFormatException ex) {
                // skip invalid values
            }
        }
        return out;
    }
}
