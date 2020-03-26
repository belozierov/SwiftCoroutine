//
//  URLSession.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 14.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Foundation.NSURLSession
import SwiftCoroutine
import AppKit

extension URLSession {
    
    public typealias DataResponse = (data: Data, response: URLResponse)
    
    @inlinable public func dataTaskFuture(for url: URL) -> CoFuture<DataResponse> {
        dataTaskFuture(for: URLRequest(url: url))
    }
    
    public func dataTaskFuture(for urlRequest: URLRequest) -> CoFuture<DataResponse> {
        let promise = CoPromise<DataResponse>()
        let task = dataTask(with: urlRequest) {
            if let error = $2 {
                promise.fail(error)
            } else if let data = $0, let response = $1 {
                promise.success((data, response))
            } else {
                promise.fail(URLError(.badServerResponse))
            }
        }
        task.resume()
        promise.whenCanceled(task.cancel)
        return promise
    }
    
//    func setThubnail2(url: URL, in imageView: NSImageView) {
//        URLSession.shared.dataTask(with: url) { data, _, error in
//            guard let image = data.flatMap(NSImage.init) else {
//                DispatchQueue.main.async {
//                    imageView.image = self.placeholder
//                }
//            }
//            DispatchQueue.global().async {
//                let resized = image.makeThubnail()
//                DispatchQueue.main.async {
//                    imageView.image = resized
//                }
//            }
//        }
//    }
//    
//    func awaitThubnail(url: URL) throws -> UIImage {
//        let (data, _, error) = try Coroutine.await {
//            URLSession.shared.dataTask(with: url, completionHandler: $0).resume()
//        }
//        guard let image = data.flatMap(UIImage.init)
//            else { throw error ?? URLError(.cannotParseResponse) }
//        return try TaskScheduler.global.await {
//            image.makeThubnail()
//        }
//    }
//
//    func setThubnail(url: URL) {
//        CoroutineDispatcher.main.execute {
//            let thubnail = try? self.awaitThubnail(url: url)
//            self.imageView.image = thubnail ?? self.placeholder
//        }
//    }
    
    func thubnail(url: URL) -> CoFuture<NSImage> {
        URLSession.shared.dataTaskFuture(for: url)
            .map { NSImage(data: $0.data) }
            .unwrap { throw URLError(.cannotParseResponse) }
            .flatMap { TaskScheduler.global.submit($0.makeThubnail) }
    }
    
//    func setThubnail(url: URL) {
//        CoroutineDispatcher.main.execute {
//            let thubnail = try? self.thubnail(url: url).await()
//            self.imageView.image = thubnail ?? self.placeholder
//        }
//    }
//
//    func setThubnail2(url: URL) {
//        thubnail(url: url).whenComplete {
//            self.imageView.image = try? $0.get ?? self.placeholder
//        }
//    }
    
}

extension NSImage {
    
    func makeThubnail() -> NSImage {
        self
    }
    
}
