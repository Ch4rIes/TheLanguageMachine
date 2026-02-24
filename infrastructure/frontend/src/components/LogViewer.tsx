interface Props {
  log: string;
}

export function LogViewer({ log }: Props) {
  return (
    <pre
      style={{
        background: '#1e1e1e',
        color: '#d4d4d4',
        borderRadius: 6,
        padding: 16,
        fontSize: 12,
        overflowY: 'auto',
        maxHeight: 320,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-all',
      }}
    >
      {log || '(empty)'}
    </pre>
  );
}
