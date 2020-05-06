//
//  CoroutineCancelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 05.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineCancelTests: XCTestCase {

    func testCancel() {
        _testCancel(DispatchQueue.global().coroutineFuture) {
            try Coroutine.delay(.seconds(1))
        }
        _testCancel(DispatchQueue.global().coroutineFuture) {
            try DispatchQueue.global().await {
                try Coroutine.delay(.seconds(1))
            }
        }
    }
    
    func testCancel2() {
        _testCancel(CoFuture.init) {
            try Coroutine.delay(.seconds(1))
        }
        _testCancel(CoFuture.init) {
            try DispatchQueue.global().await {
                try Coroutine.delay(.seconds(1))
            }
        }
    }
    
    private typealias Task = () throws -> Void
    
    private func _testCancel(_ launcher: (@escaping Task) -> CoFuture<Void>, block: @escaping Task) {
        let exp = expectation(description: "testCancel")
        exp.expectedFulfillmentCount = 10_000
        for _ in 0..<10_000 {
            let time = (0..<50).randomElement()!
            let future = launcher {
                do {
                    try block()
                    XCTFail()
                } catch let error as CoroutineError {
                    XCTAssertEqual(error, .canceled)
                } catch {
                    XCTFail()
                }
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(time), execute: future.cancel)
            future.whenCanceled { exp.fulfill() }
        }
        wait(for: [exp], timeout: 10)
    }
    
    
}
