/**
 * Turns an answer's light markdown (**bold**, _italic_, "- " lists, "[1]"
 * citations) into a small typed tree the template renders with ordinary
 * bindings. Nothing here produces HTML, so answer text — which can echo
 * retrieved content — can never inject markup.
 */
export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'italic'; text: string }
  | { kind: 'cite'; index: number };

export type Block =
  | { kind: 'paragraph'; inlines: Inline[] }
  | { kind: 'list'; items: Inline[][] };

const INLINE = /\*\*(.+?)\*\*|(?<![A-Za-z0-9])_(.+?)_(?![A-Za-z0-9])|\[(\d{1,3})\]/g;
const LIST_ITEM = /^\s*(?:[-*•]|\d+[.)])\s+(.*)$/;
const HEADING = /^#{1,6}\s+(.*)$/;

export function parseInline(text: string): Inline[] {
  const parts: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const start = match.index ?? 0;
    if (start > last) {
      parts.push({ kind: 'text', text: text.slice(last, start) });
    }
    if (match[1] !== undefined) {
      parts.push({ kind: 'bold', text: match[1] });
    } else if (match[2] !== undefined) {
      parts.push({ kind: 'italic', text: match[2] });
    } else {
      parts.push({ kind: 'cite', index: Number(match[3]) });
    }
    last = start + match[0].length;
  }
  if (last < text.length) {
    parts.push({ kind: 'text', text: text.slice(last) });
  }
  return parts;
}

export function formatAnswer(text: string): Block[] {
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  let list: Inline[][] | null = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', inlines: parseInline(paragraph.join(' ')) });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) {
      blocks.push({ kind: 'list', items: list });
      list = null;
    }
  };

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    const item = LIST_ITEM.exec(rawLine);

    if (!line) {
      flushParagraph();
      flushList();
    } else if (item) {
      flushParagraph();
      (list ??= []).push(parseInline(item[1].trim()));
    } else {
      flushList();
      const heading = HEADING.exec(line);
      if (heading) {
        flushParagraph();
        blocks.push({ kind: 'paragraph', inlines: [{ kind: 'bold', text: heading[1] }] });
      } else {
        paragraph.push(line);
      }
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}
