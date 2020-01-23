//
//  URLSession.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if os(macOS)
import Foundation.NSURLSession
#else
import FoundationNetworking
import Foundation
#endif

extension URLSession {
    
    public typealias DataResponse = (data: Data, response: URLResponse)
    
    @inlinable public func dataTaskFuture(for url: URL) -> CoFuture<DataResponse> {
        dataTaskFuture(for: URLRequest(url: url))
    }
    
    public func dataTaskFuture(for urlRequest: URLRequest) -> CoFuture<DataResponse> {
        let promise = CoPromise<DataResponse>()
        let task = dataTask(with: urlRequest) {
            if let error = $2 {
                promise.send(error)
            } else if let data = $0, let response = $1 {
                promise.send((data, response))
            } else {
                promise.send(URLError(.badServerResponse))
            }
        }
        task.resume()
        promise.addCancelHandler(execute: task.cancel)
        return promise
    }
    
}
