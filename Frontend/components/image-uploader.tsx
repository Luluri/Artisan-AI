"use client"

import type React from "react"

import { useState } from "react"
import { Upload, ImagePlus, Download, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Image from "next/image"
import { transformImage } from "@/app/actions"

export function ImageUploader() {
  const [originalImage, setOriginalImage] = useState<string | null>(null)
  const [transformedImage, setTransformedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("original")
  const [transformStyle, setTransformStyle] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setOriginalImage(event.target?.result as string)
        setTransformedImage(null)
        setTransformStyle(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleTransform = async () => {
    if (!originalImage) return

    setIsLoading(true)
    try {
      const result = await transformImage(originalImage)
      setTransformedImage(result.transformedImage)
      setTransformStyle(result.selectedStyle)
      setActiveTab("transformed")
    } catch (error) {
      console.error("Error transforming image:", error)
      alert("Failed to transform image. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!transformedImage) return

    const link = document.createElement("a")
    link.href = transformedImage
    link.download = "transformed-image.png"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <>
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Upload Your Image</CardTitle>
          <CardDescription>Select an image file to transform with our AI model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center">
            <label
              htmlFor="image-upload"
              className="w-full h-64 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer hover:bg-muted/50 transition-colors"
            >
              <ImagePlus className="h-10 w-10 text-muted-foreground mb-2" />
              <span className="text-muted-foreground">Click or drag and drop to upload</span>
              <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>
          </div>
        </CardContent>
        <CardFooter className="flex justify-center">
          <Button onClick={handleTransform} disabled={!originalImage || isLoading} className="w-full max-w-xs">
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Transforming...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Transform Image
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      {originalImage && (
        <Card>
          <CardHeader>
            <CardTitle>Image Preview</CardTitle>
            <CardDescription>Compare your original image with the AI-transformed version</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="original">Original</TabsTrigger>
                <TabsTrigger value="transformed" disabled={!transformedImage}>
                  Transformed
                </TabsTrigger>
              </TabsList>
              <TabsContent value="original" className="mt-4">
                <div className="relative aspect-square w-full overflow-hidden rounded-lg border">
                  <Image
                    src={originalImage || "/placeholder.svg"}
                    alt="Original image"
                    fill
                    className="object-contain"
                  />
                </div>
              </TabsContent>
              <TabsContent value="transformed" className="mt-4">
                {transformedImage ? (
                  <div className="space-y-4">
                    {transformStyle && (
                      <div className="bg-muted p-3 rounded-lg">
                        <p className="text-sm font-medium">
                          Style used by AI: <span className="font-bold">{transformStyle}</span>
                        </p>
                      </div>
                    )}
                    <div className="relative aspect-square w-full overflow-hidden rounded-lg border">
                      <Image
                        src={transformedImage || "/placeholder.svg"}
                        alt="Transformed image"
                        fill
                        className="object-contain"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center">
                    <p className="text-muted-foreground">Transform your image to see the result here</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
          {transformedImage && (
            <CardFooter className="flex justify-center">
              <Button onClick={handleDownload} className="w-full max-w-xs">
                <Download className="mr-2 h-4 w-4" />
                Download Transformed Image
              </Button>
            </CardFooter>
          )}
        </Card>
      )}
    </>
  )
}
