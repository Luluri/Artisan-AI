import { ImageUploader } from "../components/image-uploader";


export default function Home() {
  return (
    <div className="container mx-auto py-10 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-6">AI Image Transformer</h1>
        <p className="text-center text-muted-foreground mb-10">
          Upload an image, let the AI change is style !!! <br />
        </p>
        <ImageUploader />
      </div>
    </div>
  )
}
