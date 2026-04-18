import { useState, useRef, useCallback } from 'react';

/**
 * AudioUploader
 *
 * Drag-and-drop zone or click-to-browse file input.
 * Accepts .wav files only. Shows the selected filename and size.
 * Calls onFileSelect(file) when a valid file is chosen.
 */
export default function AudioUploader({ onFileSelect, disabled }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      const ext = file.name.split('.').pop().toLowerCase();
      if (ext !== 'wav') {
        alert('Please select a .wav file.');
        return;
      }
      setSelectedFile(file);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  const onClick = () => {
    if (!disabled && inputRef.current) inputRef.current.click();
  };

  const onChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
    e.target.value = '';  // Allow re-selecting same file
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      className={`
        card card-glow relative cursor-pointer rounded-xl
        border-2 border-dashed p-8 text-center
        transition-all duration-300 ease-out
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-saffron-500/50 hover:bg-ink-800/80'}
        ${dragOver ? 'border-saffron-400 bg-saffron-500/5 scale-[1.01]' : 'border-clay-700/40'}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".wav,audio/wav"
        onChange={onChange}
        className="hidden"
        disabled={disabled}
      />

      {/* Icon */}
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-clay-900/50 ring-1 ring-clay-700/30">
        <svg className="h-7 w-7 text-saffron-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
        </svg>
      </div>

      {selectedFile ? (
        <div className="animate-fade-in">
          <p className="text-lg font-semibold text-clay-100">{selectedFile.name}</p>
          <p className="mt-1 text-sm text-clay-600">{formatSize(selectedFile.size)}</p>
        </div>
      ) : (
        <div>
          <p className="text-lg text-clay-200">
            Drop a <span className="font-mono text-saffron-400">.wav</span> file here
          </p>
          <p className="mt-1 text-sm text-clay-600">or click to browse</p>
        </div>
      )}
    </div>
  );
}
