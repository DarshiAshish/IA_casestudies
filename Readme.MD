\documentclass{article}
\usepackage{listings}

\title{Case Studies}
\date{}

\begin{document}

\maketitle

This document outlines the four projects included in this repository, all of which use a unified environment.  
To install the necessary dependencies, execute the following command:

\begin{lstlisting}
pip install -r requirements.txt
\end{lstlisting}

\section*{Running the Servers}

Each project can be executed using the following commands:

\begin{itemize}
    \item \textbf{SMS Spam Classifier}
    \begin{lstlisting}
    python spam_words.py
    \end{lstlisting}

    \item \textbf{Text Summarization}
    \begin{lstlisting}
    python summary.py
    \end{lstlisting}

    \item \textbf{Text Classification}
    \begin{lstlisting}
    python test_model.py
    \end{lstlisting}

    \item \textbf{Sentiment Analysis}
    \begin{lstlisting}
    python sentiment.py
    \end{lstlisting}
\end{itemize}

Ensure that all required packages are installed before running the scripts.

\end{document}
