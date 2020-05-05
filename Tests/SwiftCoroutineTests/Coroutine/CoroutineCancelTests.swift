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
        _testCancel {
            try Coroutine.delay(.milliseconds(200))
        }
    }
    
    func testCancel2() {
        _testCancel {
            try DispatchQueue.global().await {
                try Coroutine.delay(.milliseconds(200))
            }
        }
    }
    
    private func _testCancel(_ block: @escaping () throws -> Void) {
        let exp = expectation(description: "testCancel2")
        exp.expectedFulfillmentCount = 10_000
        let dispatcher = DispatchQueue.global()
        for _ in 0..<10_000 {
            let time = (0..<50).randomElement()!
            let future = dispatcher.coroutineFuture {
                do {
                    try block()
                    XCTFail()
                } catch let error as CoroutineError {
                    XCTAssertEqual(error, .canceled)
                } catch {
                    XCTFail()
                }
            }
            dispatcher.asyncAfter(deadline: .now() + .milliseconds(time), execute: future.cancel)
            future.whenCanceled { exp.fulfill() }
        }
        wait(for: [exp], timeout: 10)
    }
    
    
}
