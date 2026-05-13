# Errors

## [ERR-20260512-001] read_tool_pages_parameter

**Logged**: 2026-05-12T00:00:00Z
**Priority**: low
**Status**: pending
**Area**: config

### Summary
Read tool failed when called with an empty `pages` parameter for a markdown file.

### Error
```
Invalid pages parameter: "". Use formats like "1-5", "3", or "10-20". Pages are 1-indexed.
```

### Context
- Operation attempted: Read `gateway/platforms/ADDING_A_PLATFORM.md`
- Input used: `pages: ""`
- Environment: Claude Code Read tool

### Suggested Fix
Omit the optional `pages` parameter unless reading a PDF with a real page range.

### Metadata
- Reproducible: yes
- Related Files: gateway/platforms/ADDING_A_PLATFORM.md

---
