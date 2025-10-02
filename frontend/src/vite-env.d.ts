/// <reference types="vite/client" />
declare module '*.svg?react' {
  import * as React from 'react';
  const C: React.FC<React.SVGProps<SVGSVGElement>>;
  export default C;
}

declare module '*.module.scss' {
  const classes: { [key: string]: string };
  // @ts-expect-error - этот файл используется для типизации
  export default classes;
}

declare module '*.scss' {
  const classes: { [key: string]: string };
  export default classes;
}
