import { createClient } from '@supabase/supabase-js'
import Papa from 'papaparse'

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const SHEET_ID = process.env.SHEET_ID!
const CSV_URL = `https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv`

const toNum = (val: string) => {
  const clean = val?.replace(/,/g, '').trim()
  return clean === '' || clean == null ? null : Number(clean)
}

export async function GET() {
  try {
    const response = await fetch(CSV_URL)
    if (!response.ok) throw new Error('No se pudo obtener el Sheet. Verifica que sea público.')

    const csv = await response.text()

    // Las primeras 2 filas son indicadores de festivos, la fila 3 es el header real
    const lines = csv.split('\n')
    const cleanCsv = lines.slice(2).join('\n')

    const { data, errors } = Papa.parse(cleanCsv, { header: true, skipEmptyLines: true })
    if (errors.length) throw new Error(`Error al parsear CSV: ${errors[0].message}`)

    const rows = (data as any[]).map((row) => ({
      date:             row['Date/Platform']?.trim() || null,
      total:            toNum(row['Total']),
      bamboo_co:        toNum(row['Bamboo CO']),
      bamboo_br:        toNum(row['Bamboo BR']),
      zamp_killb:       toNum(row['Zamp/Killb']),
      dlocal:           toNum(row['Dlocal']),
      local_payment:    toNum(row['Local Payment']),
      monnet_payment:   toNum(row['Monnet Payment']),
      paysend:          toNum(row['Paysend']),
      thunes:           toNum(row['Thunes']),
      rapyd:            toNum(row['Rapyd']),
      payoneer:         toNum(row['Payoneer']),
      convera:          toNum(row['Convera']),
      wise:             toNum(row['Wise']),
      more:             toNum(row['More']),
      bamboo_pe_ur:     toNum(row['Bamboo PE + UR']),
      zamp_kbi:         toNum(row['ZAMP KBI']),
      astropay:         toNum(row['AstroPay']),
      bamboo_mx:        toNum(row['Bamboo MX']),
      bamboo_pe:        toNum(row['Bamboo PE']),
      bvnk:             toNum(row['BVNK']),
      transfermate:     toNum(row['Transfermate']),
      cobre:            toNum(row['Cobre']),
      bvnk_stablecoin:  toNum(row['BVNK Stablecoin']),
      tapi:             toNum(row['Tapi']),
      zinli:            toNum(row['Zinli']),
      depayments_pix:   toNum(row['DePayments (PIX)']),
    }))

    // Borra datos anteriores y recarga frescos
    await supabase.from('prefunding').delete().neq('id', 0)

    const { error } = await supabase.from('prefunding').insert(rows)
    if (error) throw error

    return Response.json({ success: true, inserted: rows.length })
  } catch (err: any) {
    return Response.json({ error: err.message }, { status: 500 })
  }
}
