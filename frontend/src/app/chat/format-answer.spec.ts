import { formatAnswer, parseInline } from './format-answer';

describe('parseInline', () => {
  it('splits bold, italic and citations out of plain text', () => {
    expect(parseInline('See **Jersey** [1] and _more_')).toEqual([
      { kind: 'text', text: 'See ' },
      { kind: 'bold', text: 'Jersey' },
      { kind: 'text', text: ' ' },
      { kind: 'cite', index: 1 },
      { kind: 'text', text: ' and ' },
      { kind: 'italic', text: 'more' }
    ]);
  });

  it('does not treat snake_case words as italics', () => {
    expect(parseInline('use my_var_name here')).toEqual([{ kind: 'text', text: 'use my_var_name here' }]);
  });

  it('keeps html-looking text as plain text', () => {
    expect(parseInline('<img src=x onerror=alert(1)>')).toEqual([
      { kind: 'text', text: '<img src=x onerror=alert(1)>' }
    ]);
  });
});

describe('formatAnswer', () => {
  it('builds paragraphs and bullet lists', () => {
    const blocks = formatAnswer("Here's what I found:\n\n**Shirt** [1]\n- Price: 10\n- Size: M\n\nDone.");

    expect(blocks.map((b) => b.kind)).toEqual(['paragraph', 'paragraph', 'list', 'paragraph']);
    const list = blocks[2];
    expect(list.kind === 'list' && list.items.length).toBe(2);
  });

  it('joins wrapped lines of one paragraph', () => {
    const blocks = formatAnswer('first line\nsecond line');

    expect(blocks).toEqual([{ kind: 'paragraph', inlines: [{ kind: 'text', text: 'first line second line' }] }]);
  });

  it('renders markdown headings as bold paragraphs', () => {
    expect(formatAnswer('## Title')).toEqual([{ kind: 'paragraph', inlines: [{ kind: 'bold', text: 'Title' }] }]);
  });

  it('returns nothing for empty text', () => {
    expect(formatAnswer('  \n ')).toEqual([]);
  });
});
