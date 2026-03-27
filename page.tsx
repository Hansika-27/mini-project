'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, Send, Minus, Wheat, Soup, Flame, Salad, Drumstick, TrendingDown } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_WASTE_API_URL ?? 'http://localhost:5000'

export default function PredictPage() {
  const [formData, setFormData] = useState({
    date: '',
    dayOfWeek: 'Monday',
    mealType: 'lunch',
    studentsEnrolled: '',
    averageAttendance: '',
    specialEvent: 'no',
    weather: 'clear',
    holidayPeriod: 'no',
    menusServed: '',
    leftoverFromPreviousDay: '',
    menuItems: [] as string[]
  })

  const [prediction, setPrediction] = useState<{
    predictedWaste:  number
    cost:            number
    recommendation:  string
    confidence:      number
    recommendations: {
      quantities: {
        rice_kg:    number
        dal_kg:     number
        roti_count: number
        veg_kg:     number
        nonveg_kg:  number
      }
      reduction_percent:  number
      current_waste_kg:   number
      optimized_waste_kg: number
      waste_reduction_kg: number
    }
  } | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleMenusChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0
    setFormData(prev => ({
      ...prev,
      menusServed: String(value),
      menuItems: Array(value).fill('')
    }))
  }

  const handleMenuItemChange = (index: number, value: string) => {
    const updated = [...formData.menuItems]
    updated[index] = value
    setFormData(prev => ({ ...prev, menuItems: updated }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const payload = {
        studentsEnrolled:        formData.studentsEnrolled,
        averageAttendance:       formData.averageAttendance,
        menusServed:             formData.menusServed,
        leftoverFromPreviousDay: formData.leftoverFromPreviousDay || '0',
        specialEvent:            formData.specialEvent,
        mealType:                formData.mealType,
        dayOfWeek:               formData.dayOfWeek,
        menuItems:               formData.menuItems,
      }

      const response = await fetch(`${API_URL}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body?.error ?? `API error ${response.status}`)
      }

      const data = await response.json()

      setPrediction({
        predictedWaste:  data.predicted_waste_kg,
        cost:            data.cost,
        recommendation:  data.recommendation,
        confidence:      data.confidence,
        recommendations: data.recommendations,
      })

    } catch (err: any) {
      setError(err.message ?? 'Could not reach the prediction API. Is the server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-emerald-50 py-12">
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Link href="/" className="p-2 hover:bg-white rounded-lg transition-colors">
            <ArrowLeft className="w-6 h-6 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Food Wastage Predictor</h1>
            <p className="text-gray-600">Enter hostel details for accurate waste predictions</p>
          </div>
        </div>

        {/* Form */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Date</label>
                <input type="date" name="date" value={formData.date}
                  onChange={handleChange} required
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Day of Week</label>
                <select name="dayOfWeek" value={formData.dayOfWeek} onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500">
                  {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].map(d => (
                    <option key={d}>{d}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Meal Type</label>
                <select name="mealType" value={formData.mealType} onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg">
                  <option value="breakfast">Breakfast</option>
                  <option value="lunch">Lunch</option>
                  <option value="dinner">Dinner</option>
                  <option value="snacks">Snacks</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Students Enrolled</label>
                <input type="number" name="studentsEnrolled" value={formData.studentsEnrolled}
                  onChange={handleChange} required placeholder="e.g., 500"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                />
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Average Attendance (%)</label>
                <input type="number" name="averageAttendance" value={formData.averageAttendance}
                  onChange={handleChange} required min="0" max="100" placeholder="e.g., 85"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Special Event</label>
                <select name="specialEvent" value={formData.specialEvent} onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg">
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Weather</label>
                <select name="weather" value={formData.weather} onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg">
                  <option value="clear">Clear</option>
                  <option value="cloudy">Cloudy</option>
                  <option value="rainy">Rainy</option>
                  <option value="hot">Hot</option>
                  <option value="cold">Cold</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Holiday Period</label>
                <select name="holidayPeriod" value={formData.holidayPeriod} onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg">
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Number of Menus Served</label>
                <input type="number" value={formData.menusServed}
                  onChange={handleMenusChange} placeholder="e.g., 3"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Leftover from Previous Day (kg)</label>
                <input type="number" name="leftoverFromPreviousDay" value={formData.leftoverFromPreviousDay}
                  onChange={handleChange} placeholder="e.g., 5"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                />
              </div>
            </div>

            {formData.menuItems.length > 0 && (
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-gray-700">Menu Items</label>
                {formData.menuItems.map((item, index) => (
                  <select key={index} value={item}
                    onChange={(e) => handleMenuItemChange(index, e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg">
                    <option value="">Select Menu Item</option>
                    <option value="rice">Rice</option>
                    <option value="dal">Dal</option>
                    <option value="veg">Main Veg</option>
                    <option value="nonveg">Main Non-Veg</option>
                  </select>
                ))}
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                ⚠️ {error}
              </div>
            )}

            <button type="submit" disabled={loading}
              className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-400 text-white font-bold py-4 rounded-lg flex items-center justify-center gap-2 text-lg transition-colors">
              {loading ? (
                <><span className="inline-block animate-spin"><Minus className="w-5 h-5" /></span> Analyzing...</>
              ) : (
                <><Send className="w-5 h-5" /> Get Prediction</>
              )}
            </button>

          </form>
        </div>

        {/* Results */}
        {prediction && (
          <div className="space-y-6">

            {/* Summary Card */}
            <div className="bg-white rounded-2xl shadow-lg p-8 border-2 border-emerald-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Prediction Results</h2>

              {/* 4 KPI tiles */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-gradient-to-br from-emerald-50 to-transparent p-5 rounded-xl border border-emerald-200">
                  <p className="text-xs text-gray-500 font-medium mb-1">Predicted Waste</p>
                  <p className="text-2xl font-bold text-emerald-600">{prediction.predictedWaste} kg</p>
                </div>
                <div className="bg-gradient-to-br from-blue-50 to-transparent p-5 rounded-xl border border-blue-200">
                  <p className="text-xs text-gray-500 font-medium mb-1">Cost Impact</p>
                  <p className="text-2xl font-bold text-blue-600">₹{prediction.cost}</p>
                </div>
                <div className="bg-gradient-to-br from-amber-50 to-transparent p-5 rounded-xl border border-amber-200">
                  <p className="text-xs text-gray-500 font-medium mb-1">Risk Level</p>
                  <p className={`text-2xl font-bold ${
                    prediction.recommendation === 'High'   ? 'text-red-600' :
                    prediction.recommendation === 'Medium' ? 'text-amber-600' :
                    'text-green-600'
                  }`}>{prediction.recommendation}</p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-transparent p-5 rounded-xl border border-purple-200">
                  <p className="text-xs text-gray-500 font-medium mb-1">Model Accuracy (R²)</p>
                  <p className="text-2xl font-bold text-purple-600">{prediction.confidence}%</p>
                </div>
              </div>

              {/* Original vs Optimized Waste */}
              <div className="bg-gray-50 rounded-xl p-5 space-y-3">
                <p className="text-sm font-semibold text-gray-700 mb-2">Waste Comparison</p>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 text-sm">Original Predicted Waste</span>
                  <span className="font-bold text-red-600">{prediction.recommendations.current_waste_kg} kg</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 text-sm">Optimized Predicted Waste</span>
                  <span className="font-bold text-emerald-600">{prediction.recommendations.optimized_waste_kg} kg</span>
                </div>
                <div className="border-t pt-3 flex justify-between items-center">
                  <span className="text-gray-700 font-semibold">Waste Reduction</span>
                  <span className="font-bold text-teal-600 text-lg flex items-center gap-1">
                    <TrendingDown className="w-5 h-5" />
                    {prediction.recommendations.waste_reduction_kg} kg saved
                  </span>
                </div>
              </div>
            </div>

            {/* Recommended Quantities Card */}
            <div className="bg-white rounded-2xl shadow-lg p-8 border-2 border-teal-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-1">Recommended Cooking Quantities</h2>
              <p className="text-sm text-gray-500 mb-6">
                Reduced by {prediction.recommendations.reduction_percent}% based on predicted waste level
              </p>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[
                  { label: 'Rice',    value: `${prediction.recommendations.quantities.rice_kg} kg`,   Icon: Wheat },
                  { label: 'Dal',     value: `${prediction.recommendations.quantities.dal_kg} kg`,    Icon: Soup },
                  { label: 'Roti',    value: `${prediction.recommendations.quantities.roti_count}`,   Icon: Flame },
                  { label: 'Veg',     value: `${prediction.recommendations.quantities.veg_kg} kg`,    Icon: Salad },
                  { label: 'Non-Veg', value: `${prediction.recommendations.quantities.nonveg_kg} kg`, Icon: Drumstick },
                ].map(({ label, value, Icon }) => (
                  <div key={label} className="bg-teal-50 border border-teal-200 rounded-xl p-4 text-center">
                    <div className="flex justify-center mb-2">
                      <Icon className="w-6 h-6 text-teal-500" />
                    </div>
                    <p className="text-xs text-gray-500 font-medium mb-1">{label}</p>
                    <p className="text-lg font-bold text-teal-700">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button onClick={() => setPrediction(null)}
                className="flex-1 px-6 py-3 border-2 border-emerald-600 text-emerald-600 rounded-lg hover:bg-emerald-50 transition-colors font-semibold">
                Make Another Prediction
              </button>
              <Link href="/dashboard"
                className="flex-1 px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors font-semibold text-center">
                View Dashboard
              </Link>
            </div>

          </div>
        )}
      </div>
    </main>
  )
}
