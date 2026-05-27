"use client"

import { useState } from "react"

export default function Home() {

  const [result, setResult] = useState<null | {
    gravidade: string
    recuperacao: number
    recorrencia: string
  }>(null)

  // CLASSIFICAÇÃO DE GRAVIDADE
 
  function classificarGravidade(dias: number) {

    if (dias <= 14) {
      return "Leve"
    }

    if (dias <= 30) {
      return "Moderada"
    }

    if (dias <= 90) {
      return "Grave"
    }

    return "Severa"
  }

  // CORES DINÂMICAS
 
  function corGravidade(gravidade: string) {

    switch (gravidade) {

      case "Leve":
        return "text-green-400"

      case "Moderada":
        return "text-yellow-400"

      case "Grave":
        return "text-orange-400"

      case "Severa":
        return "text-red-500"

      default:
        return "text-white"
    }
  }

  function corCard(gravidade: string) {

    switch (gravidade) {

      case "Leve":
        return "border-green-500"

      case "Moderada":
        return "border-yellow-400"

      case "Grave":
        return "border-orange-400"

      case "Severa":
        return "border-red-500"

      default:
        return "border-slate-700"
    }
  }

  function corBarra(gravidade: string) {

    switch (gravidade) {

      case "Leve":
        return "bg-green-500"

      case "Moderada":
        return "bg-yellow-400"

      case "Grave":
        return "bg-orange-400"

      case "Severa":
        return "bg-red-500"

      default:
        return "bg-slate-500"
    }
  }

  // RESULTADO TEMPORARIO (RANDOM) - NECESSARIO SUBSTITUIR

  function handlePrediction(
    e: React.FormEvent<HTMLFormElement>
  ) {

    e.preventDefault()

    // MOCK TEMPORÁRIO
    // Depois será substituído pelo backend real

    const recuperacao =
      Math.floor(Math.random() * 120) + 1

    const gravidade =
      classificarGravidade(recuperacao)

    const recorrencia =
      recuperacao > 60
        ? "Alto"
        : recuperacao > 30
        ? "Moderado"
        : "Baixo"

    setResult({
      gravidade,
      recuperacao,
      recorrencia
    })
  }

  return (

    <main className="min-h-screen bg-slate-950 text-white">

      {/* NAVBAR */}

      <nav className="border-b border-slate-800 px-10 py-6">

        <div className="max-w-7xl mx-auto">

          <h1 className="text-3xl font-bold text-green-400">
            InjuryPredict
          </h1>

          <p className="text-slate-400 text-sm mt-1">
            Predição inteligente de gravidade de lesões no futebol europeu
          </p>

        </div>

      </nav>

      {/* SUBTITULO */}

      <section className="max-w-7xl mx-auto px-6 pt-14">

        <h1 className="text-5xl font-bold leading-tight max-w-4xl">
          Sistema de Predição de Gravidade de Lesões
        </h1>

        <p className="text-slate-400 text-lg mt-6 max-w-3xl">
          Plataforma baseada em análise preditiva para
          estimativa de gravidade, tempo de recuperação
          e risco de recorrência em atletas profissionais.
        </p>

      </section>
      {/* MÉTRICAS */}

      <section className="max-w-7xl mx-auto px-6 py-10">

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">

            <p className="text-slate-400 mb-2">
              Registros analisados
            </p>

            <h3 className="text-4xl font-bold text-green-400">
              15.603
            </h3>

          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">

            <p className="text-slate-400 mb-2">
              Modelo utilizado
            </p>

            <h3 className="text-3xl font-bold text-green-400">
              Random Forest
            </h3>

          </div>

        </div>

      </section>
      {/* CONTEÚDO */}

      <section className="max-w-7xl mx-auto px-6 pb-16 grid grid-cols-1 lg:grid-cols-2 gap-10">

        {/* FORMULÁRIO */}

        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8">

          <h2 className="text-3xl font-bold mb-8">
            Nova Predição
          </h2>

          <form
            onSubmit={handlePrediction}
            className="space-y-6"
          >

            {/* IDADE */}
            <div>

              <label className="block mb-2 text-slate-300">
                Idade do jogador
              </label>

              <input
                type="number"
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-3"
                placeholder="Ex: 27"
                required
              />

            </div>

            {/* POSIÇÃO */}
            <div>

              <label className="block mb-2 text-slate-300">
                Posição
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-3"
                required
              >

                <option>Atacante</option>
                <option>Meio-campo</option>
                <option>Zagueiro</option>
                <option>Lateral</option>
                <option>Goleiro</option>

              </select>

            </div>

            {/* LIGA */}
            <div>

              <label className="block mb-2 text-slate-300">
                Liga
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-3"
                required
              >

                <option>Premier League</option>
                <option>La Liga</option>
                <option>Bundesliga</option>
                <option>Serie A</option>
                <option>Ligue 1</option>

              </select>

            </div>

            {/* BOTÃO */}
            <button
              type="submit"
              className="w-full bg-green-500 hover:bg-green-400 transition text-black font-bold py-4 rounded-xl"
            >
              Gerar análise preditiva
            </button>

          </form>

        </div>

        {/* RESULTADO */}

        <div
          className={`bg-slate-900 border rounded-3xl p-8 flex flex-col justify-center transition-all duration-500 ${result ? corCard(result.gravidade) : "border-slate-800"}`}
        >

          {!result ? (

            <div>

              <h2 className="text-3xl font-bold mb-4">
                Resultado da Predição
              </h2>

              <p className="text-slate-400">
                Preencha o formulário para gerar
                a análise preditiva da lesão.
              </p>

            </div>

          ) : (

            <div>

              <h2 className="text-3xl font-bold mb-10">
                Resultado da Predição
              </h2>

              <div className="space-y-10">

                {/* GRAVIDADE */}
                <div>

                  <p className="text-slate-400 mb-2">
                    Gravidade estimada
                  </p>

                  <h3
                    className={`text-6xl font-bold ${corGravidade(result.gravidade)}`}
                  >
                    {result.gravidade}
                  </h3>

                </div>

                {/* BARRA */}
                <div>

                  <p className="text-slate-400 mb-3">
                    Escala de gravidade
                  </p>

                  <div className="w-full h-5 bg-slate-800 rounded-full overflow-hidden">

                    <div
                      className={`h-full transition-all duration-700 ${corBarra(result.gravidade)}`}
                      style={{
                        width:
                          result.gravidade === "Leve"
                            ? "25%"
                            : result.gravidade === "Moderada"
                            ? "50%"
                            : result.gravidade === "Grave"
                            ? "75%"
                            : "100%"
                      }}
                    />

                  </div>

                </div>

                {/* RECUPERAÇÃO */}
                <div>

                  <p className="text-slate-400 mb-2">
                    Tempo estimado de recuperação
                  </p>

                  <h3 className="text-5xl font-bold">
                    {result.recuperacao} dias
                  </h3>

                </div>

                {/* RECORRÊNCIA */}
                <div>

                  <p className="text-slate-400 mb-2">
                    Risco de recorrência
                  </p>

                  <h3 className="text-3xl font-bold">
                    {result.recorrencia}
                  </h3>

                </div>

                {/* INSIGHTS */}
                <div className="bg-slate-950 border border-slate-800 rounded-2xl p-5">

                  <h3 className="text-xl font-bold mb-3">
                    Insights Médicos
                  </h3>

                  <p className="text-slate-400 leading-relaxed">

                    {result.gravidade === "Leve" &&
                      "Lesão de baixa complexidade clínica com recuperação rápida e baixo impacto competitivo."
                    }

                    {result.gravidade === "Moderada" &&
                      "Lesão com necessidade de monitoramento físico e retorno progressivo às atividades."
                    }

                    {result.gravidade === "Grave" &&
                      "Caso de maior impacto esportivo com necessidade de recuperação prolongada."
                    }

                    {result.gravidade === "Severa" &&
                      "Lesão crítica com longo período de afastamento e elevado risco de recorrência."
                    }

                  </p>

                </div>

              </div>

            </div>

          )}

        </div>

      </section>

    </main>

  )
}